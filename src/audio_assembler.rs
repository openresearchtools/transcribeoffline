#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeakerSpan {
    pub speaker: String,
    pub start_sample: u64,
    pub end_sample: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptPiece {
    pub start_sample: u64,
    pub end_sample: u64,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssignedPiece {
    pub speaker: Option<String>,
    pub start_sample: u64,
    pub end_sample: u64,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeakerTurn {
    pub speaker: String,
    pub start_sample: u64,
    pub end_sample: u64,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct AssembleOptions {
    pub nearest_tolerance_samples: u64,
    pub alignment_offset_samples: i64,
}

impl Default for AssembleOptions {
    fn default() -> Self {
        Self {
            nearest_tolerance_samples: 1600,
            alignment_offset_samples: 0,
        }
    }
}

fn overlap_len(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> u64 {
    let start = a_start.max(b_start);
    let end = a_end.min(b_end);
    end.saturating_sub(start)
}

fn distance_to_span(piece: &TranscriptPiece, span: &SpeakerSpan) -> u64 {
    if piece.end_sample <= span.start_sample {
        span.start_sample - piece.end_sample
    } else if span.end_sample <= piece.start_sample {
        piece.start_sample - span.end_sample
    } else {
        0
    }
}

fn shift_sample(sample: u64, offset_samples: i64) -> u64 {
    if offset_samples >= 0 {
        sample.saturating_add(offset_samples as u64)
    } else {
        sample.saturating_sub(offset_samples.unsigned_abs())
    }
}

fn shift_piece(piece: &TranscriptPiece, offset_samples: i64) -> TranscriptPiece {
    TranscriptPiece {
        start_sample: shift_sample(piece.start_sample, offset_samples),
        end_sample: shift_sample(piece.end_sample, offset_samples),
        text: piece.text.clone(),
    }
}

fn dominant_speaker_for_piece(
    piece: &TranscriptPiece,
    spans: &[SpeakerSpan],
    options: &AssembleOptions,
) -> Option<String> {
    let shifted_piece = shift_piece(piece, options.alignment_offset_samples);
    let mut best_overlap = 0u64;
    let mut best_speaker: Option<&str> = None;
    for span in spans {
        let overlap = overlap_len(
            shifted_piece.start_sample,
            shifted_piece.end_sample,
            span.start_sample,
            span.end_sample,
        );
        if overlap > best_overlap {
            best_overlap = overlap;
            best_speaker = Some(span.speaker.as_str());
        }
    }
    if best_overlap > 0 {
        return best_speaker.map(str::to_string);
    }

    let mut nearest: Option<(&SpeakerSpan, u64)> = None;
    for span in spans {
        let gap = distance_to_span(&shifted_piece, span);
        if gap <= options.nearest_tolerance_samples {
            match nearest {
                Some((_, best_gap)) if gap >= best_gap => {}
                _ => nearest = Some((span, gap)),
            }
        }
    }
    nearest.map(|(span, _)| span.speaker.clone())
}

fn first_sentence_boundary(text: &str) -> Option<usize> {
    let trimmed = text.trim();
    let offset = text.find(trimmed)?;
    for (idx, ch) in trimmed.char_indices() {
        if matches!(ch, '.' | ';' | '?' | '!') {
            let split = offset + idx + ch.len_utf8();
            let right = text[split..].trim();
            if !right.is_empty() {
                return Some(split);
            }
        }
    }
    None
}

fn trim_leading_token(text: &str) -> &str {
    text.trim_start_matches(char::is_whitespace)
}

fn first_token(text: &str) -> &str {
    trim_leading_token(text)
        .split_whitespace()
        .next()
        .unwrap_or("")
}

fn starts_with_continuation(text: &str) -> bool {
    let trimmed = trim_leading_token(text);
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if first.is_ascii_lowercase() {
        return true;
    }
    matches!(first, ',' | '\'' | '"' | '-' | ')' | ']' | ':')
}

fn starts_with_sentence_start(text: &str) -> bool {
    let trimmed = trim_leading_token(text);
    let Some(first) = trimmed.chars().next() else {
        return false;
    };
    first.is_ascii_uppercase()
}

fn ends_sentence(text: &str) -> bool {
    let trimmed = text.trim_end();
    matches!(trimmed.chars().last(), Some('.' | ';' | '?' | '!'))
}

fn word_ends_sentence(text: &str) -> bool {
    let trimmed = text.trim_end_matches(char::is_whitespace);
    matches!(trimmed.chars().last(), Some('.' | ';' | '?' | '!'))
}

fn next_piece_looks_like_continuation(next_text: &str) -> bool {
    if starts_with_continuation(next_text) {
        return true;
    }
    first_token(next_text) == "I"
}

fn split_piece_at(
    piece: &AssignedPiece,
    split_index: usize,
) -> Option<(AssignedPiece, AssignedPiece)> {
    if split_index == 0 || split_index >= piece.text.len() {
        return None;
    }

    let left_text = piece.text[..split_index].trim_end().to_string();
    let right_text = piece.text[split_index..].trim_start().to_string();
    if left_text.is_empty() || right_text.is_empty() {
        return None;
    }

    let total_len = left_text.len() + right_text.len();
    if total_len == 0 {
        return None;
    }
    let duration = piece.end_sample.saturating_sub(piece.start_sample);
    let left_duration = duration.saturating_mul(left_text.len() as u64) / total_len as u64;
    let mid = piece.start_sample.saturating_add(left_duration);

    Some((
        AssignedPiece {
            speaker: piece.speaker.clone(),
            start_sample: piece.start_sample,
            end_sample: mid.max(piece.start_sample),
            text: left_text,
        },
        AssignedPiece {
            speaker: piece.speaker.clone(),
            start_sample: mid.max(piece.start_sample),
            end_sample: piece.end_sample.max(mid),
            text: right_text,
        },
    ))
}

fn split_piece_at_sentence_boundary_with_words(
    piece: &AssignedPiece,
    words: &[TranscriptPiece],
) -> Option<(AssignedPiece, AssignedPiece)> {
    let overlapping: Vec<&TranscriptPiece> = words
        .iter()
        .filter(|word| {
            overlap_len(
                piece.start_sample,
                piece.end_sample,
                word.start_sample,
                word.end_sample,
            ) > 0
        })
        .collect();
    if overlapping.len() < 2 {
        return None;
    }

    let boundary_index = overlapping.iter().enumerate().find_map(|(idx, word)| {
        (idx + 1 < overlapping.len() && word_ends_sentence(&word.text)).then_some(idx)
    })?;

    let left_text = overlapping[..=boundary_index]
        .iter()
        .map(|word| word.text.trim())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let right_text = overlapping[boundary_index + 1..]
        .iter()
        .map(|word| word.text.trim())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if left_text.is_empty() || right_text.is_empty() {
        return None;
    }

    let split_sample = overlapping[boundary_index].end_sample;
    Some((
        AssignedPiece {
            speaker: piece.speaker.clone(),
            start_sample: piece.start_sample,
            end_sample: split_sample.max(piece.start_sample),
            text: left_text,
        },
        AssignedPiece {
            speaker: piece.speaker.clone(),
            start_sample: split_sample.max(piece.start_sample),
            end_sample: piece.end_sample.max(split_sample),
            text: right_text,
        },
    ))
}

fn previous_confirmed_speaker(pieces: &[AssignedPiece]) -> Option<String> {
    pieces.iter().rev().find_map(|piece| piece.speaker.clone())
}

fn previous_confirmed_piece(pieces: &[AssignedPiece]) -> Option<&AssignedPiece> {
    pieces.iter().rev().find(|piece| piece.speaker.is_some())
}

fn next_confirmed_piece(pieces: &[AssignedPiece], start_index: usize) -> Option<&AssignedPiece> {
    pieces
        .get(start_index + 1..)
        .and_then(|rest| rest.iter().find(|piece| piece.speaker.is_some()))
}

fn assigned_piece_word_count(piece: &AssignedPiece) -> usize {
    piece.text.split_whitespace().count()
}

fn assign_word_speakers(
    spans: &[SpeakerSpan],
    words: &[TranscriptPiece],
    options: &AssembleOptions,
) -> Vec<AssignedPiece> {
    words
        .iter()
        .map(|word| AssignedPiece {
            speaker: dominant_speaker_for_piece(word, spans, options),
            start_sample: word.start_sample,
            end_sample: word.end_sample,
            text: word.text.clone(),
        })
        .collect()
}

fn assigned_word_duration_samples(word: &AssignedPiece) -> u64 {
    let duration = word.end_sample.saturating_sub(word.start_sample);
    duration.max(1)
}

fn assigned_word_speaker(word: &AssignedPiece) -> &str {
    word.speaker.as_deref().unwrap_or("UNASSIGNED")
}

fn dominant_speaker_for_assigned_words(words: &[AssignedPiece], begin: usize, end: usize) -> Option<String> {
    let mut totals: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for word in &words[begin..end] {
        let Some(speaker) = word.speaker.as_ref() else {
            continue;
        };
        *totals.entry(speaker.clone()).or_default() += assigned_word_duration_samples(word);
    }
    totals.into_iter().max_by_key(|(_, total)| *total).map(|(speaker, _)| speaker)
}

fn reassign_word_range(words: &mut [AssignedPiece], begin: usize, end: usize, speaker: &str) {
    for word in &mut words[begin..end] {
        word.speaker = Some(speaker.to_string());
    }
}

fn smooth_sentence_word_speakers(words: &mut [AssignedPiece]) {
    if words.is_empty() {
        return;
    }

    #[derive(Clone)]
    struct SpeakerRun {
        begin: usize,
        end: usize,
        speaker: Option<String>,
        duration_samples: u64,
    }

    fn runs_for_sentence(words: &[AssignedPiece], sentence_begin: usize, sentence_end: usize) -> Vec<SpeakerRun> {
        let mut runs = Vec::new();
        let mut i = sentence_begin;
        while i < sentence_end {
            let speaker = words[i].speaker.clone();
            let mut j = i + 1;
            while j < sentence_end && words[j].speaker == speaker {
                j += 1;
            }
            let duration_samples = words[i..j]
                .iter()
                .map(assigned_word_duration_samples)
                .sum::<u64>();
            runs.push(SpeakerRun {
                begin: i,
                end: j,
                speaker,
                duration_samples,
            });
            i = j;
        }
        runs
    }

    fn process_sentence(words: &mut [AssignedPiece], sentence_begin: usize, sentence_end: usize) {
        if sentence_begin >= sentence_end {
            return;
        }

        let mut runs = runs_for_sentence(words, sentence_begin, sentence_end);
        for idx in 1..runs.len() {
            let run = &runs[idx];
            let prev = &runs[idx - 1];
            if run.speaker == prev.speaker {
                continue;
            }
            let run_words = run.end - run.begin;
            if run_words > 4 || run.duration_samples > 20_000 {
                continue;
            }
            if !starts_with_continuation(&words[run.begin].text) {
                continue;
            }
            if let Some(prev_speaker) = prev.speaker.as_deref() {
                reassign_word_range(words, run.begin, run.end, prev_speaker);
            }
        }

        runs = runs_for_sentence(words, sentence_begin, sentence_end);
        for idx in 0..runs.len().saturating_sub(1) {
            let run = &runs[idx];
            let next = &runs[idx + 1];
            if run.speaker == next.speaker {
                continue;
            }
            let run_words = run.end - run.begin;
            if run_words > 4 || run.duration_samples > 20_000 {
                continue;
            }
            let sentence_start = run.begin == sentence_begin
                || word_ends_sentence(&words[run.begin - 1].text);
            if !sentence_start || !starts_with_sentence_start(&words[run.begin].text) {
                continue;
            }
            let next_words = next.end - next.begin;
            if next_words < run_words + 2 && next.duration_samples < run.duration_samples + 6_400 {
                continue;
            }
            if let Some(next_speaker) = next.speaker.as_deref() {
                reassign_word_range(words, run.begin, run.end, next_speaker);
            }
        }

        if let Some(best_speaker) = dominant_speaker_for_assigned_words(words, sentence_begin, sentence_end) {
            let mut total = 0u64;
            let mut best = 0u64;
            let mut second = 0u64;
            let mut counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
            for word in &words[sentence_begin..sentence_end] {
                let Some(speaker) = word.speaker.as_ref() else {
                    continue;
                };
                let dur = assigned_word_duration_samples(word);
                total += dur;
                let entry = counts.entry(speaker.clone()).or_default();
                *entry += dur;
            }
            for (speaker, weight) in counts {
                if speaker == best_speaker {
                    best = weight;
                } else if weight > second {
                    second = weight;
                }
            }
            if total > 0 {
                let clear_majority = best * 100 >= total * 60
                    || best >= second.saturating_mul(2)
                    || ((sentence_end - sentence_begin) <= 4 && best * 100 >= total * 50);
                if clear_majority {
                    reassign_word_range(words, sentence_begin, sentence_end, &best_speaker);
                }
            }
        }
    }

    let mut sentence_begin = 0usize;
    for i in 0..words.len() {
        if word_ends_sentence(&words[i].text) {
            process_sentence(words, sentence_begin, i + 1);
            sentence_begin = i + 1;
        }
    }
    process_sentence(words, sentence_begin, words.len());
}

#[derive(Clone)]
struct SentenceRange {
    begin: usize,
    end: usize,
    start_sample: u64,
    end_sample: u64,
    duration_samples: u64,
    num_words: usize,
    dominant_speaker: String,
}

fn previous_span_speaker_near(
    spans: &[SpeakerSpan],
    anchor_sample: u64,
    max_gap_samples: u64,
) -> (String, u64) {
    let mut best_speaker = None::<String>;
    let mut best_gap = u64::MAX;
    for span in spans {
        if span.end_sample > anchor_sample {
            continue;
        }
        let gap = anchor_sample.saturating_sub(span.end_sample);
        if gap <= max_gap_samples && gap < best_gap {
            best_gap = gap;
            best_speaker = Some(span.speaker.clone());
        }
    }
    (best_speaker.unwrap_or_else(|| "UNASSIGNED".to_string()), best_gap)
}

fn next_span_speaker_near(
    spans: &[SpeakerSpan],
    anchor_sample: u64,
    max_gap_samples: u64,
) -> (String, u64) {
    let mut best_speaker = None::<String>;
    let mut best_gap = u64::MAX;
    for span in spans {
        if span.start_sample < anchor_sample {
            continue;
        }
        let gap = span.start_sample.saturating_sub(anchor_sample);
        if gap <= max_gap_samples && gap < best_gap {
            best_gap = gap;
            best_speaker = Some(span.speaker.clone());
        }
    }
    (best_speaker.unwrap_or_else(|| "UNASSIGNED".to_string()), best_gap)
}

fn covering_or_previous_span_end_for_speaker_near(
    spans: &[SpeakerSpan],
    speaker: &str,
    anchor_sample: u64,
    max_gap_samples: u64,
) -> Option<(u64, u64)> {
    if speaker.is_empty() || speaker == "UNASSIGNED" {
        return None;
    }

    let mut best_end = 0u64;
    let mut best_gap = u64::MAX;
    let mut found = false;
    for span in spans {
        if span.speaker != speaker {
            continue;
        }
        let gap = if anchor_sample >= span.start_sample && anchor_sample <= span.end_sample {
            0
        } else if span.end_sample <= anchor_sample {
            anchor_sample.saturating_sub(span.end_sample)
        } else {
            continue;
        };

        if gap <= max_gap_samples && (!found || gap < best_gap || (gap == best_gap && span.end_sample > best_end)) {
            best_gap = gap;
            best_end = span.end_sample;
            found = true;
        }
    }

    found.then_some((best_end, best_gap))
}

fn next_span_start_for_speaker_near(
    spans: &[SpeakerSpan],
    speaker: &str,
    anchor_sample: u64,
    max_gap_samples: u64,
) -> Option<(u64, u64)> {
    if speaker.is_empty() || speaker == "UNASSIGNED" {
        return None;
    }

    let mut best_start = u64::MAX;
    let mut best_gap = u64::MAX;
    for span in spans {
        if span.speaker != speaker || span.start_sample < anchor_sample {
            continue;
        }
        let gap = span.start_sample.saturating_sub(anchor_sample);
        if gap <= max_gap_samples && (gap < best_gap || (gap == best_gap && span.start_sample < best_start)) {
            best_gap = gap;
            best_start = span.start_sample;
        }
    }

    (best_start != u64::MAX).then_some((best_start, best_gap))
}

fn refine_sentence_word_boundaries(words: &mut [AssignedPiece], spans: &[SpeakerSpan]) {
    if words.is_empty() || spans.is_empty() {
        return;
    }

    const BOUNDARY_GAP_SAMPLES: u64 = 3_840;
    const SHORT_SENTENCE_WORDS: usize = 6;
    const SHORT_SENTENCE_DURATION_SAMPLES: u64 = 28_000;
    const PREV_CARRY_GAP_SAMPLES: u64 = 3_840;
    const NEXT_SPAN_DELAY_SAMPLES: u64 = 3_200;
    const NEXT_SPAN_SEARCH_SAMPLES: u64 = 12_800;

    let mut ranges = Vec::new();
    let mut sentence_begin = 0usize;
    for i in 0..words.len() {
        if !word_ends_sentence(&words[i].text) && i + 1 != words.len() {
            continue;
        }

        let sentence_end = i + 1;
        let start_sample = words[sentence_begin].start_sample;
        let end_sample = words[sentence_end - 1].end_sample.max(start_sample);
        ranges.push(SentenceRange {
            begin: sentence_begin,
            end: sentence_end,
            start_sample,
            end_sample,
            duration_samples: end_sample.saturating_sub(start_sample),
            num_words: sentence_end - sentence_begin,
            dominant_speaker: dominant_speaker_for_assigned_words(words, sentence_begin, sentence_end)
                .unwrap_or_else(|| "UNASSIGNED".to_string()),
        });
        sentence_begin = sentence_end;
    }

    for i in 0..ranges.len() {
        let short_sentence = ranges[i].num_words <= SHORT_SENTENCE_WORDS
            || ranges[i].duration_samples <= SHORT_SENTENCE_DURATION_SAMPLES;
        if !short_sentence {
            continue;
        }

        if !starts_with_sentence_start(&words[ranges[i].begin].text) {
            continue;
        }

        let prev_sentence_speaker = if i > 0 {
            ranges[i - 1].dominant_speaker.clone()
        } else {
            "UNASSIGNED".to_string()
        };
        let next_sentence_speaker = if i + 1 < ranges.len() {
            ranges[i + 1].dominant_speaker.clone()
        } else {
            "UNASSIGNED".to_string()
        };

        let (prev_span_speaker, prev_gap_samples) =
            previous_span_speaker_near(spans, ranges[i].start_sample, BOUNDARY_GAP_SAMPLES);
        let (next_span_speaker, next_gap_samples) =
            next_span_speaker_near(spans, ranges[i].end_sample, BOUNDARY_GAP_SAMPLES);

        if prev_sentence_speaker != "UNASSIGNED"
            && prev_sentence_speaker == next_sentence_speaker
            && ranges[i].dominant_speaker != prev_sentence_speaker
        {
            reassign_word_range(words, ranges[i].begin, ranges[i].end, &prev_sentence_speaker);
            ranges[i].dominant_speaker = prev_sentence_speaker;
            continue;
        }

        if prev_sentence_speaker != "UNASSIGNED"
            && prev_sentence_speaker == prev_span_speaker
            && ranges[i].dominant_speaker != prev_sentence_speaker
            && next_span_speaker == ranges[i].dominant_speaker
            && prev_gap_samples <= BOUNDARY_GAP_SAMPLES
        {
            reassign_word_range(words, ranges[i].begin, ranges[i].end, &prev_sentence_speaker);
            ranges[i].dominant_speaker = prev_sentence_speaker;
            continue;
        }

        if next_sentence_speaker != "UNASSIGNED"
            && next_sentence_speaker == next_span_speaker
            && ranges[i].dominant_speaker != next_sentence_speaker
            && prev_span_speaker == ranges[i].dominant_speaker
            && next_gap_samples <= BOUNDARY_GAP_SAMPLES
        {
            reassign_word_range(words, ranges[i].begin, ranges[i].end, &next_sentence_speaker);
            ranges[i].dominant_speaker = next_sentence_speaker;
            continue;
        }

        if i > 0
            && prev_sentence_speaker != "UNASSIGNED"
            && ranges[i].dominant_speaker != "UNASSIGNED"
            && ranges[i].dominant_speaker != prev_sentence_speaker
            && (next_sentence_speaker == ranges[i].dominant_speaker
                || next_sentence_speaker == "UNASSIGNED")
        {
            let has_prev_same = covering_or_previous_span_end_for_speaker_near(
                spans,
                &prev_sentence_speaker,
                ranges[i].start_sample,
                PREV_CARRY_GAP_SAMPLES,
            );
            let has_next_dom = next_span_start_for_speaker_near(
                spans,
                &ranges[i].dominant_speaker,
                ranges[i].start_sample,
                NEXT_SPAN_SEARCH_SAMPLES,
            );
            if let (Some((_, prev_same_gap_samples)), Some((next_dom_start_sample, _))) =
                (has_prev_same, has_next_dom)
            {
                if prev_same_gap_samples <= PREV_CARRY_GAP_SAMPLES
                    && next_dom_start_sample.saturating_sub(ranges[i].start_sample)
                        >= NEXT_SPAN_DELAY_SAMPLES
                {
                    reassign_word_range(words, ranges[i].begin, ranges[i].end, &prev_sentence_speaker);
                    ranges[i].dominant_speaker = prev_sentence_speaker;
                }
            }
        }
    }
}

fn speaker_from_assigned_words(
    piece: &TranscriptPiece,
    assigned_words: &[AssignedPiece],
) -> Option<String> {
    let mut totals: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for word in assigned_words {
        if overlap_len(piece.start_sample, piece.end_sample, word.start_sample, word.end_sample) == 0 {
            continue;
        }
        let Some(speaker) = word.speaker.as_ref() else {
            continue;
        };
        *totals.entry(speaker.clone()).or_default() += assigned_word_duration_samples(word);
    }
    totals.into_iter().max_by_key(|(_, total)| *total).map(|(speaker, _)| speaker)
}

pub fn assign_pieces(
    spans: &[SpeakerSpan],
    pieces: &[TranscriptPiece],
    words: &[TranscriptPiece],
    options: &AssembleOptions,
) -> Vec<AssignedPiece> {
    let mut assigned_words = assign_word_speakers(spans, words, options);
    smooth_sentence_word_speakers(&mut assigned_words);
    refine_sentence_word_boundaries(&mut assigned_words, spans);
    pieces
        .iter()
        .map(|piece| AssignedPiece {
            speaker: speaker_from_assigned_words(piece, &assigned_words)
                .or_else(|| dominant_speaker_for_piece(piece, spans, options)),
            start_sample: piece.start_sample,
            end_sample: piece.end_sample,
            text: piece.text.clone(),
        })
        .collect()
}

pub fn repair_unassigned_boundaries_with_words(
    pieces: &mut Vec<AssignedPiece>,
    words: &[TranscriptPiece],
) {
    let source = pieces.clone();
    let mut repaired = Vec::with_capacity(source.len() + 2);

    for (index, piece) in source.iter().enumerate() {
        if piece.speaker.is_some() {
            repaired.push(piece.clone());
            continue;
        }

        let prev_speaker = previous_confirmed_speaker(&repaired);
        let prev_piece = previous_confirmed_piece(&repaired);
        let next_piece = next_confirmed_piece(&source, index);
        let next_speaker = next_piece.and_then(|p| p.speaker.clone());

        if let (Some(prev), Some(next)) = (prev_speaker.clone(), next_speaker.clone()) {
            if prev == next {
                let mut reassigned = piece.clone();
                reassigned.speaker = Some(prev);
                repaired.push(reassigned);
                continue;
            }
        }

        if let Some(prev) = prev_speaker.clone() {
            if starts_with_continuation(&piece.text) {
                if let Some(split_index) = first_sentence_boundary(&piece.text) {
                    let split = split_piece_at_sentence_boundary_with_words(piece, words)
                        .or_else(|| split_piece_at(piece, split_index));
                    if let Some((mut left, mut right)) = split {
                        left.speaker = Some(prev.clone());
                        if let Some(next) = next_speaker.clone() {
                            if starts_with_sentence_start(&right.text) {
                                right.speaker = Some(next);
                            } else {
                                right.speaker = Some(prev);
                            }
                        } else {
                            right.speaker = Some(prev);
                        }
                        repaired.push(left);
                        repaired.push(right);
                        continue;
                    }
                }

                let mut reassigned = piece.clone();
                reassigned.speaker = Some(prev);
                repaired.push(reassigned);
                continue;
            }
        }

        if let Some(next) = next_speaker.clone() {
            let prev_ended_sentence = prev_piece
                .map(|piece| ends_sentence(&piece.text))
                .unwrap_or(true);
            if prev_ended_sentence
                && starts_with_sentence_start(&piece.text)
                && first_token(&piece.text) != "I"
            {
                let mut reassigned = piece.clone();
                reassigned.speaker = Some(next);
                repaired.push(reassigned);
                continue;
            }
        }

        if let Some(next) = next_speaker.clone() {
            let next_text = next_piece.map(|p| p.text.as_str()).unwrap_or("");
            if !ends_sentence(&piece.text) && next_piece_looks_like_continuation(next_text) {
                let mut reassigned = piece.clone();
                reassigned.speaker = Some(next);
                repaired.push(reassigned);
                continue;
            }
        }

        repaired.push(piece.clone());
    }

    *pieces = repaired;
}

pub fn repair_unassigned_boundaries(pieces: &mut Vec<AssignedPiece>) {
    repair_unassigned_boundaries_with_words(pieces, &[]);
}

pub fn absorb_short_speaker_islands(pieces: &mut [AssignedPiece]) {
    if pieces.len() < 3 {
        return;
    }

    for index in 1..pieces.len() - 1 {
        let Some(cur_speaker) = pieces[index].speaker.clone() else {
            continue;
        };
        let Some(prev_speaker) = pieces[index - 1].speaker.clone() else {
            continue;
        };
        let Some(next_speaker) = pieces[index + 1].speaker.clone() else {
            continue;
        };

        if prev_speaker != next_speaker || cur_speaker == prev_speaker {
            continue;
        }

        let duration_samples = pieces[index]
            .end_sample
            .saturating_sub(pieces[index].start_sample);
        let word_count = assigned_piece_word_count(&pieces[index]);
        let short_fragment = duration_samples <= 32_000
            && (word_count <= 4
                || starts_with_continuation(&pieces[index].text)
                || !ends_sentence(&pieces[index].text));
        if !short_fragment {
            continue;
        }

        pieces[index].speaker = Some(prev_speaker);
    }
}

fn should_insert_space_between(left: &str, right: &str) -> bool {
    if left.is_empty() || right.is_empty() {
        return false;
    }
    let right_trimmed = trim_leading_token(right);
    let Some(first) = right_trimmed.chars().next() else {
        return false;
    };
    !matches!(
        first,
        '.' | ',' | ';' | '?' | '!' | '\'' | '"' | ':' | ')' | ']' | '}'
    )
}

pub fn merge_turns(pieces: &[AssignedPiece]) -> Vec<SpeakerTurn> {
    let mut turns: Vec<SpeakerTurn> = Vec::new();

    for piece in pieces {
        let speaker = piece
            .speaker
            .clone()
            .unwrap_or_else(|| "UNASSIGNED".to_string());
        let text = piece.text.trim();
        if text.is_empty() {
            continue;
        }

        if let Some(last) = turns.last_mut() {
            if last.speaker == speaker {
                if !last.text.ends_with(' ') && should_insert_space_between(&last.text, text) {
                    last.text.push(' ');
                }
                last.text.push_str(text);
                last.end_sample = piece.end_sample.max(last.end_sample);
                continue;
            }
        }

        turns.push(SpeakerTurn {
            speaker,
            start_sample: piece.start_sample,
            end_sample: piece.end_sample,
            text: text.to_string(),
        });
    }

    turns
}

fn merge_adjacent_turns(turns: &[SpeakerTurn]) -> Vec<SpeakerTurn> {
    let mut merged: Vec<SpeakerTurn> = Vec::new();
    for turn in turns {
        let text = turn.text.trim();
        if text.is_empty() {
            continue;
        }

        if let Some(last) = merged.last_mut() {
            if last.speaker == turn.speaker {
                if !last.text.ends_with(' ') && should_insert_space_between(&last.text, text) {
                    last.text.push(' ');
                }
                last.text.push_str(text);
                last.end_sample = last.end_sample.max(turn.end_sample);
                continue;
            }
        }

        merged.push(SpeakerTurn {
            speaker: turn.speaker.clone(),
            start_sample: turn.start_sample,
            end_sample: turn.end_sample,
            text: text.to_string(),
        });
    }
    merged
}

fn speaker_turn_word_count(turn: &SpeakerTurn) -> usize {
    turn.text.split_whitespace().count()
}

fn absorb_short_turn_islands(turns: &mut [SpeakerTurn]) {
    if turns.len() < 3 {
        return;
    }

    for index in 1..turns.len() - 1 {
        let cur_speaker = turns[index].speaker.clone();
        let prev_speaker = turns[index - 1].speaker.clone();
        let next_speaker = turns[index + 1].speaker.clone();

        if prev_speaker != next_speaker || cur_speaker == prev_speaker {
            continue;
        }

        let duration_samples = turns[index]
            .end_sample
            .saturating_sub(turns[index].start_sample);
        let word_count = speaker_turn_word_count(&turns[index]);
        let short_fragment = duration_samples <= 32_000
            && (word_count <= 4
                || starts_with_continuation(&turns[index].text)
                || !ends_sentence(&turns[index].text));
        if !short_fragment {
            continue;
        }

        turns[index].speaker = prev_speaker;
    }
}

fn collapse_unassigned_tail_turn(pieces: &[AssignedPiece]) -> Option<SpeakerTurn> {
    let mut text = String::new();
    let mut start_sample = None;
    let mut end_sample = 0u64;

    for piece in pieces {
        let trimmed = piece.text.trim();
        if trimmed.is_empty() {
            continue;
        }

        if start_sample.is_none() {
            start_sample = Some(piece.start_sample);
        }
        if !text.is_empty() && should_insert_space_between(&text, trimmed) {
            text.push(' ');
        }
        text.push_str(trimmed);
        end_sample = end_sample.max(piece.end_sample);
    }

    start_sample.map(|start_sample| SpeakerTurn {
        speaker: "UNASSIGNED".to_string(),
        start_sample,
        end_sample: end_sample.max(start_sample),
        text,
    })
}

fn tail_should_follow_previous_speaker(previous_turn: &SpeakerTurn, tail_turn: &SpeakerTurn) -> bool {
    let tail_word_count = speaker_turn_word_count(tail_turn);
    let tail_duration_samples = tail_turn.end_sample.saturating_sub(tail_turn.start_sample);

    if tail_word_count <= 2 && tail_duration_samples <= 12_800 {
        return true;
    }

    if starts_with_continuation(&tail_turn.text) {
        return true;
    }

    !ends_sentence(&previous_turn.text)
        && !starts_with_sentence_start(&tail_turn.text)
        && tail_word_count <= 6
        && tail_duration_samples <= 24_000
}

pub fn assemble_turns_with_words(
    spans: &[SpeakerSpan],
    pieces: &[TranscriptPiece],
    words: &[TranscriptPiece],
    options: &AssembleOptions,
) -> Vec<SpeakerTurn> {
    let mut assigned = assign_pieces(spans, pieces, words, options);
    repair_unassigned_boundaries_with_words(&mut assigned, words);
    absorb_short_speaker_islands(&mut assigned);

    let stable_until = spans.iter().map(|span| span.end_sample).max().unwrap_or(0);
    let split_index = assigned
        .iter()
        .position(|piece| piece.start_sample >= stable_until)
        .unwrap_or(assigned.len());
    let (stable_assigned, tail_assigned) = assigned.split_at(split_index);

    let mut turns = merge_turns(stable_assigned);
    if let Some(mut tail) = collapse_unassigned_tail_turn(tail_assigned) {
        if let Some(previous_turn) = turns.last() {
            if tail_should_follow_previous_speaker(previous_turn, &tail) {
                tail.speaker = previous_turn.speaker.clone();
            }
        }
        turns.push(tail);
    }
    absorb_short_turn_islands(&mut turns);
    merge_adjacent_turns(&turns)
}

pub fn turns_to_markdown(turns: &[SpeakerTurn], sample_rate_hz: u32) -> String {
    fn fmt_time(sample: u64, sample_rate_hz: u32) -> String {
        if sample_rate_hz == 0 {
            return "00:00:00.000".to_string();
        }
        let total_ms = sample.saturating_mul(1000) / sample_rate_hz as u64;
        let hours = total_ms / 3_600_000;
        let minutes = (total_ms % 3_600_000) / 60_000;
        let seconds = (total_ms % 60_000) / 1000;
        let millis = total_ms % 1000;
        format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
    }

    let mut out = String::new();
    for turn in turns {
        if !out.is_empty() {
            out.push('\n');
            out.push('\n');
        }
        out.push_str("### ");
        out.push_str(&turn.speaker);
        out.push(' ');
        out.push('[');
        out.push_str(&fmt_time(turn.start_sample, sample_rate_hz));
        out.push_str(" - ");
        out.push_str(&fmt_time(turn.end_sample, sample_rate_hz));
        out.push(']');
        out.push('\n');
        out.push_str(turn.text.trim());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_trailing_live_text_as_single_unassigned_turn() {
        let spans = vec![SpeakerSpan {
            speaker: "SPEAKER_00".to_string(),
            start_sample: 0,
            end_sample: 32_000,
        }];
        let pieces = vec![
            TranscriptPiece {
                start_sample: 0,
                end_sample: 16_000,
                text: "Hello there.".to_string(),
            },
            TranscriptPiece {
                start_sample: 32_000,
                end_sample: 48_000,
                text: "This keeps going".to_string(),
            },
            TranscriptPiece {
                start_sample: 48_000,
                end_sample: 64_000,
                text: "after diarization lag.".to_string(),
            },
        ];

        let turns = assemble_turns_with_words(&spans, &pieces, &[], &AssembleOptions::default());
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker, "SPEAKER_00");
        assert_eq!(turns[1].speaker, "UNASSIGNED");
        assert_eq!(turns[1].text, "This keeps going after diarization lag.");
    }

    #[test]
    fn keeps_short_preview_tail_on_previous_speaker_sentence() {
        let spans = vec![SpeakerSpan {
            speaker: "SPEAKER_00".to_string(),
            start_sample: 0,
            end_sample: 32_000,
        }];
        let pieces = vec![
            TranscriptPiece {
                start_sample: 0,
                end_sample: 30_000,
                text: "They sort of say one thing and then contradict themselves within two"
                    .to_string(),
            },
            TranscriptPiece {
                start_sample: 32_000,
                end_sample: 34_000,
                text: "seconds.".to_string(),
            },
        ];

        let turns = assemble_turns_with_words(&spans, &pieces, &[], &AssembleOptions::default());
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker, "SPEAKER_00");
        assert!(turns[0].text.ends_with("within two seconds."));
    }

    #[test]
    fn formats_live_markdown_timestamps_like_offline_transcripts() {
        let turns = vec![SpeakerTurn {
            speaker: "SPEAKER_00".to_string(),
            start_sample: 1_000,
            end_sample: 17_250,
            text: "Hello.".to_string(),
        }];

        let markdown = turns_to_markdown(&turns, 1_000);
        assert!(markdown.contains("### SPEAKER_00 [00:00:01.000 - 00:00:17.250]"));
    }

    #[test]
    fn absorbs_tiny_turn_island_between_same_speaker() {
        let mut turns = vec![
            SpeakerTurn {
                speaker: "SPEAKER_00".to_string(),
                start_sample: 0,
                end_sample: 40_000,
                text: "I really".to_string(),
            },
            SpeakerTurn {
                speaker: "SPEAKER_01".to_string(),
                start_sample: 40_000,
                end_sample: 41_000,
                text: "enjoyed".to_string(),
            },
            SpeakerTurn {
                speaker: "SPEAKER_00".to_string(),
                start_sample: 41_000,
                end_sample: 80_000,
                text: "it.".to_string(),
            },
        ];

        absorb_short_turn_islands(&mut turns);
        let merged = merge_adjacent_turns(&turns);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].speaker, "SPEAKER_00");
        assert_eq!(merged[0].text, "I really enjoyed it.");
    }

    #[test]
    fn reassigns_short_sentence_to_surrounding_speaker() {
        let spans = vec![
            SpeakerSpan {
                speaker: "SPEAKER_00".to_string(),
                start_sample: 0,
                end_sample: 16_000,
            },
            SpeakerSpan {
                speaker: "SPEAKER_01".to_string(),
                start_sample: 16_000,
                end_sample: 32_000,
            },
            SpeakerSpan {
                speaker: "SPEAKER_00".to_string(),
                start_sample: 32_000,
                end_sample: 56_000,
            },
        ];
        let words = vec![
            TranscriptPiece {
                start_sample: 0,
                end_sample: 4_000,
                text: "This".to_string(),
            },
            TranscriptPiece {
                start_sample: 4_000,
                end_sample: 8_000,
                text: "works.".to_string(),
            },
            TranscriptPiece {
                start_sample: 16_000,
                end_sample: 20_000,
                text: "Today".to_string(),
            },
            TranscriptPiece {
                start_sample: 20_000,
                end_sample: 24_000,
                text: "I".to_string(),
            },
            TranscriptPiece {
                start_sample: 24_000,
                end_sample: 28_000,
                text: "agree.".to_string(),
            },
            TranscriptPiece {
                start_sample: 32_000,
                end_sample: 36_000,
                text: "Still".to_string(),
            },
            TranscriptPiece {
                start_sample: 36_000,
                end_sample: 40_000,
                text: "works.".to_string(),
            },
        ];
        let pieces = vec![
            TranscriptPiece {
                start_sample: 0,
                end_sample: 8_000,
                text: "This works.".to_string(),
            },
            TranscriptPiece {
                start_sample: 16_000,
                end_sample: 28_000,
                text: "Today I agree.".to_string(),
            },
            TranscriptPiece {
                start_sample: 32_000,
                end_sample: 40_000,
                text: "Still works.".to_string(),
            },
        ];

        let turns = assemble_turns_with_words(&spans, &pieces, &words, &AssembleOptions::default());
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker, "SPEAKER_00");
        assert!(turns[0].text.contains("Today I agree."));
    }
}
