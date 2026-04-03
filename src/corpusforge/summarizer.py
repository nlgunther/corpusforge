"""
CorpusForge Summarizer
=======================
LLM-powered text analysis via Google Gemini (gemini-2.5-flash).

Design notes:
  - google.genai is imported lazily inside __init__ so the rest of the CLI
    works without it installed or configured. Commands that don't need the
    LLM (status, files, search, cluster without --name-topics) stay instant.
  - All three public methods share a single private _generate() helper that
    owns the Gemini call, error handling, and availability check. Adding a
    fourth LLM operation (e.g. propose_merge in Phase 4) is a 5-line wrapper.
  - Failure contract: every public method returns a safe fallback string on
    any failure. They never raise — LLM unavailability must not block any
    pipeline operation.

Requires: GEMINI_API_KEY environment variable.
"""

import json


class CorpusSummarizer:
    """
    LLM text analysis for CorpusForge.

    Usage:
        summarizer = CorpusSummarizer()

        # File summary (returns formatted string):
        summary = summarizer.summarize_file(content_preview="...", title="My Doc")

        # Topic naming (returns {"name": str, "description": str}):
        info = summarizer.name_topic(["chunk text 1", "chunk text 2"])

        # Topic compilation (returns Markdown string):
        md = summarizer.compile_topic("Auth Flow", "desc", ["chunk1", "chunk2"])
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

        # Lazy import: fail gracefully rather than crashing the whole CLI when
        # google-genai is absent or GEMINI_API_KEY is not set.
        try:
            from google import genai
            self.client = genai.Client()
            self._available = True
        except ImportError:
            self.client = None
            self._available = False
            print(
                "Warning: google-genai is not installed. "
                "LLM features will be skipped. "
                "Run `pip install google-genai` and set GEMINI_API_KEY to enable."
            )
        except Exception as e:
            self.client = None
            self._available = False
            print(f"Warning: Gemini client failed to initialize: {e}. LLM features disabled.")

    # ───────────────────────────────────────────────────────────────────
    # Private helper — single source of truth for all Gemini calls
    # ───────────────────────────────────────────────────────────────────

    def _generate(
        self,
        system: str,
        prompt: str,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        """
        Make a single Gemini API call and return the response text.

        All public methods delegate here. This is the only place that
        touches self.client, handles exceptions, and checks _available.

        Args:
            system:      system instruction string.
            prompt:      user prompt string.
            temperature: 0.0–1.0. Use low values (~0.1–0.2) for factual
                         extraction; higher (~0.4) for narrative generation.
            json_mode:   if True, sets response_mime_type="application/json"
                         to force the model to return valid JSON.

        Returns the response text on success, or an empty string on failure.
        The empty string signals callers to use their own fallback value.
        """
        if not self._available:
            return ""

        from google.genai import types

        config_kwargs: dict = dict(
            system_instruction=system,
            temperature=temperature,
        )
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return response.text
        except Exception as e:
            print(f"Warning: Gemini call failed. Error: {e}")
            return ""

    # ───────────────────────────────────────────────────────────────────
    # Public methods
    # ───────────────────────────────────────────────────────────────────

    def summarize_file(
        self,
        content_preview: str,
        title: str | None = None,
        user_summary: str | None = None,
    ) -> str:
        """
        Generate a 2-3 sentence summary + key topics from a content preview.

        Intended for use during ingest (cf ingest --summarize). The preview
        is typically the first ~1500 words of the document.

        Returns a formatted string on success, or a placeholder on failure.
        Never raises — failures must not block the ingest pipeline.
        """
        system = (
            "You are a meticulous technical editor analyzing a document corpus. "
            "Your task is to review the provided document preview and generate:\n"
            "1. A concise 2-3 sentence summary capturing the core technical or structural claims.\n"
            "2. A bulleted list of 3-5 key topics discussed.\n"
            "Do not use flowery language. Be direct and objective."
        )
        prompt = f"Document Title: {title or 'Unknown'}\n"
        if user_summary:
            prompt += f"User's Context Note: {user_summary}\n"
        prompt += f"\nDocument Content Preview:\n{content_preview}\n"

        result = self._generate(system, prompt, temperature=0.2)
        return result or "Auto-summary unavailable (LLM call failed or not configured)."

    def name_topic(self, chunk_texts: list[str]) -> dict:
        """
        Infer a topic name and description from a sample of cluster chunks.

        Used by TopicEngine.name_topics() — called once per topic after
        cf cluster --name-topics.

        Args:
            chunk_texts: list of chunk content strings (typically 3 samples).

        Returns {"name": str, "description": str}. Falls back to safe
        defaults if the LLM is unavailable or returns malformed JSON.
        """
        system = (
            "You are an expert taxonomist and technical editor. Read the following text snippets "
            "which have been mathematically clustered together. Your job is to determine their common theme.\n"
            "Return a JSON object with exactly two keys:\n"
            "1. 'name': A short, 2-4 word title for the topic.\n"
            "2. 'description': A concise 1-sentence summary of what these snippets discuss.\n"
            "Do not output markdown formatting, only the raw JSON."
        )
        prompt = "Cluster Snippets:\n\n" + "\n\n---\n\n".join(chunk_texts)

        raw = self._generate(system, prompt, temperature=0.1, json_mode=True)
        if not raw:
            return {"name": "Unnamed Topic", "description": "LLM unavailable or skipped."}

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"Warning: Topic naming returned invalid JSON: {raw[:80]}")
            return {"name": "Unnamed Topic", "description": "Error parsing LLM response."}

    def compile_topic(
        self,
        topic_name: str,
        topic_description: str,
        chunk_texts: list[str],
    ) -> str:
        """
        Weave a list of related chunks into a single cohesive Markdown document.

        Note: this is the LLM-based compilation path, retained for cases where
        prose output is desired. For most use cases, prefer the local outline
        and linear-export formats produced by LocalCompiler, which require no
        API call and have no chunk-count limit.

        Args:
            topic_name:        display name of the topic.
            topic_description: one-sentence description.
            chunk_texts:       list of chunk content strings (caller should cap
                               at ~12 to avoid context bloat and slow responses).

        Returns a Markdown string, or a placeholder on failure.
        """
        system = (
            "You are an expert technical writer and synthesizing editor. "
            "Your task is to take a set of disconnected text snippets that all relate to a single topic, "
            "and weave them together into a single, eloquently flowing, cohesive narrative document.\n"
            "Remove redundancies, organize the concepts logically with headings, and ensure smooth transitions. "
            "Output the result as beautifully formatted Markdown. "
            "Do not include introductory conversational text."
        )
        prompt = (
            f"Topic: {topic_name}\nDescription: {topic_description}\n\n"
            f"Source Snippets:\n\n" + "\n\n---\n\n".join(chunk_texts)
        )

        result = self._generate(system, prompt, temperature=0.4)
        return result or "Compilation failed (LLM unavailable or not configured)."

    def propose_merge(self, text_a: str, text_b: str) -> str:
        """
        Drafts a unified, merged text from two overlapping chunks.

        This uses the LLM as an automated editor to identify the unique facts 
        in both chunks and weave them together into a single, cohesive paragraph 
        without losing any granular detail.

        Args:
            text_a: The first chunk's content.
            text_b: The second chunk's content.

        Returns:
            A string containing the LLM's proposed merged text.
        """
        system = (
            "You are a meticulous technical editor and archivist. "
            "You will be given two text snippets from a personal knowledge base that have been "
            "mathematically identified as overlapping or highly similar.\n\n"
            "Your task is to merge them into a single, cohesive snippet. "
            "You must:\n"
            "1. Remove all redundant phrasing.\n"
            "2. Preserve EVERY unique technical fact, detail, and piece of context from BOTH snippets.\n"
            "3. Ensure the resulting text flows naturally as a single thought.\n\n"
            "Output ONLY the merged text. Do not include markdown blocks, pleasantries, or explanations."
        )
        
        prompt = f"--- Snippet A ---\n{text_a}\n\n--- Snippet B ---\n{text_b}"

        # We use a lower temperature (0.2) because we want factual consolidation, 
        # not creative hallucination.
        result = self._generate(system, prompt, temperature=0.2)
        return result or "Error: LLM could not generate a merge proposal."

    # Once chunk_topics are populated, a local summary can be assembled from:
    #   - The document title and heading_path outline
    #   - Top 3 topics by relevance (from the file_topics view)
    #   - Proportion of code vs. text chunks
    # Proposed interface:
    #   summarize_file(..., strategy="local" | "llm" | "auto")
    # where "auto" uses local if topics exist, falls back to LLM if not.
