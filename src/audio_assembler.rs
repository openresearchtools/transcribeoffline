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

fn split_piece_at(piece: &AssignedPiece, split_index: usize) -> Option<(AssignedPiece, AssignedPiece)> {
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

pub fn assign_pieces(
    spans: &[SpeakerSpan],
    pieces: &[TranscriptPiece],
    options: &AssembleOptions,
) -> Vec<AssignedPiece> {
    pieces
        .iter()
        .map(|piece| AssignedPiece {
            speaker: dominant_speaker_for_piece(piece, spans, options),
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

pub fn assemble_turns_with_words(
    spans: &[SpeakerSpan],
    pieces: &[TranscriptPiece],
    words: &[TranscriptPiece],
    options: &AssembleOptions,
) -> Vec<SpeakerTurn> {
    let mut assigned = assign_pieces(spans, pieces, options);
    repair_unassigned_boundaries_with_words(&mut assigned, words);
    merge_turns(&assigned)
}

pub fn turns_to_markdown(turns: &[SpeakerTurn], sample_rate_hz: u32) -> String {
    fn fmt_time(sample: u64, sample_rate_hz: u32) -> String {
        if sample_rate_hz == 0 {
            return "00:00.000".to_string();
        }
        let total_ms = sample.saturating_mul(1000) / sample_rate_hz as u64;
        let minutes = total_ms / 60_000;
        let seconds = (total_ms % 60_000) / 1000;
        let millis = total_ms % 1000;
        format!("{minutes:02}:{seconds:02}.{millis:03}")
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
