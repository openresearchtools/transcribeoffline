use crate::audio_assembler::{
    assemble_turns_with_words, turns_to_markdown, AssembleOptions, SpeakerSpan, SpeakerTurn,
    TranscriptPiece,
};
use crate::bridge::{
    AudioSessionEvent, AUDIO_EVENT_DIARIZATION_SPAN_COMMIT, AUDIO_EVENT_FLAG_PREVIEW,
    AUDIO_EVENT_FLAG_SNAPSHOT_END, AUDIO_EVENT_FLAG_SNAPSHOT_START,
    AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT, AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON,
    AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT,
};

#[derive(Clone, Debug)]
pub struct OrchestratorSnapshot {
    pub markdown: String,
    pub latest_transcription_json: Option<String>,
}

pub struct DiarizedTranscriptOrchestrator {
    sample_rate_hz: u32,
    options: AssembleOptions,
    final_spans: Vec<SpeakerSpan>,
    preview_spans: Vec<SpeakerSpan>,
    pending_preview_spans: Vec<SpeakerSpan>,
    pieces: Vec<TranscriptPiece>,
    words: Vec<TranscriptPiece>,
    turns: Vec<SpeakerTurn>,
    markdown: String,
    latest_transcription_json: Option<String>,
}

impl DiarizedTranscriptOrchestrator {
    pub fn new(sample_rate_hz: u32) -> Self {
        Self {
            sample_rate_hz,
            options: AssembleOptions::default(),
            final_spans: Vec::new(),
            preview_spans: Vec::new(),
            pending_preview_spans: Vec::new(),
            pieces: Vec::new(),
            words: Vec::new(),
            turns: Vec::new(),
            markdown: String::new(),
            latest_transcription_json: None,
        }
    }

    pub fn ingest_event(&mut self, event: &AudioSessionEvent) -> bool {
        let mut changed = false;
        match event.kind {
            AUDIO_EVENT_DIARIZATION_SPAN_COMMIT => {
                let speaker = if !event.text.trim().is_empty() {
                    event.text.trim().to_string()
                } else if event.speaker_id >= 0 {
                    format!("SPEAKER_{:02}", event.speaker_id)
                } else {
                    "UNASSIGNED".to_string()
                };
                let span = SpeakerSpan {
                    speaker,
                    start_sample: event.start_sample,
                    end_sample: event.end_sample,
                };
                if (event.flags & AUDIO_EVENT_FLAG_PREVIEW) != 0 {
                    if (event.flags & AUDIO_EVENT_FLAG_SNAPSHOT_START) != 0 {
                        self.pending_preview_spans.clear();
                    }
                    insert_unique_span(&mut self.pending_preview_spans, span);
                    if (event.flags & AUDIO_EVENT_FLAG_SNAPSHOT_END) != 0
                        && self.preview_spans != self.pending_preview_spans
                    {
                        self.preview_spans = self.pending_preview_spans.clone();
                        changed = self.final_spans.is_empty();
                    }
                } else {
                    changed |= insert_unique_span(&mut self.final_spans, span);
                }
            }
            AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT => {
                changed |= insert_unique_piece(
                    &mut self.pieces,
                    TranscriptPiece {
                        start_sample: event.start_sample,
                        end_sample: event.end_sample,
                        text: event.text.clone(),
                    },
                );
            }
            AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT => {
                changed |= insert_unique_piece(
                    &mut self.words,
                    TranscriptPiece {
                        start_sample: event.start_sample,
                        end_sample: event.end_sample,
                        text: event.text.clone(),
                    },
                );
            }
            AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON => {
                self.latest_transcription_json = Some(event.text.clone());
            }
            _ => {}
        }

        if changed && !self.active_spans().is_empty() && !self.pieces.is_empty() {
            self.turns = assemble_turns_with_words(
                self.active_spans(),
                &self.pieces,
                &self.words,
                &self.options,
            );
            self.markdown = turns_to_markdown(&self.turns, self.sample_rate_hz);
        }

        changed
    }

    pub fn snapshot(&self) -> OrchestratorSnapshot {
        OrchestratorSnapshot {
            markdown: self.markdown.clone(),
            latest_transcription_json: self.latest_transcription_json.clone(),
        }
    }

    fn active_spans(&self) -> &[SpeakerSpan] {
        if !self.final_spans.is_empty() {
            &self.final_spans
        } else {
            &self.preview_spans
        }
    }
}

fn insert_unique_span(spans: &mut Vec<SpeakerSpan>, span: SpeakerSpan) -> bool {
    if spans.iter().any(|existing| {
        existing.speaker == span.speaker
            && existing.start_sample == span.start_sample
            && existing.end_sample == span.end_sample
    }) {
        return false;
    }
    spans.push(span);
    spans.sort_by_key(|item| (item.start_sample, item.end_sample));
    true
}

fn insert_unique_piece(pieces: &mut Vec<TranscriptPiece>, piece: TranscriptPiece) -> bool {
    if pieces.iter().any(|existing| {
        existing.start_sample == piece.start_sample
            && existing.end_sample == piece.end_sample
            && existing.text == piece.text
    }) {
        return false;
    }
    pieces.push(piece);
    pieces.sort_by_key(|item| (item.start_sample, item.end_sample));
    true
}
