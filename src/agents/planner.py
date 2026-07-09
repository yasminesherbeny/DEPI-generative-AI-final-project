"""
src/agents/planner.py
=====================
Planner Agent for the DEPI Generative-AI Product Description System.

The agent orchestrates the full RAG + generation pipeline in four explicit
planning steps:

    STEP 1 ─ PARSE      Extract product name and color from the input.
    STEP 2 ─ RETRIEVE   Search the FAISS vector index for the most similar
                        products already in the catalogue (uses the
                        retrieve_for_prompt() wrapper in ProductRetriever,
                        which was designed specifically for this agent).
    STEP 3 ─ ENRICH     If the user left the color blank, infer it from the
                        top retrieved result so the model gets a complete prompt.
    STEP 4 ─ GENERATE   Feed the (possibly enriched) product name + color into
                        the fine-tuned LoRA GPT-2 model and get a description.

The agent returns a PlanResult that includes:
  • the final generated description
  • the top similar products found by retrieval (useful as a reference)
  • a step-by-step plan trace for transparency / debugging

The retrieval step is optional: if the FAISS index is not yet built
(run scripts/build_index.py to create it), the agent falls back gracefully
to generation only with a warning.

Usage (CLI):
    python -m src.agents.planner --name "Floral Maxi Dress" --color "Blue"
    python -m src.agents.planner --name "Leather Wallet"          # color inferred
    python -m src.agents.planner --interactive                     # loop

Usage (module):
    from src.agents.planner import PlannerAgent
    agent = PlannerAgent()
    result = agent.run("Running Shoes", "Black")
    print(result.description)
    print(result.similar_products)   # list of retrieved catalogue items
"""

from __future__ import annotations

import argparse
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── lazy imports so the agent can be imported even before all deps are ready ──
_retriever_available = True
try:
    from src.retrieval.retriever import ProductRetriever
except Exception:
    _retriever_available = False

from src.inference.infer import ProductDescriptionGenerator


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class PlanStep:
    """A single step recorded in the agent's plan trace."""
    name: str
    status: str            # "ok" | "skipped" | "warn"
    detail: str = ""
    duration_ms: float = 0.0

    def __str__(self) -> str:
        timing = f"  ({self.duration_ms:.0f} ms)" if self.duration_ms else ""
        return f"[{self.status.upper():7}] {self.name}{timing}: {self.detail}"


@dataclass
class PlanResult:
    """
    Output of PlannerAgent.run().

    Attributes
    ----------
    name, color     : final product name and color used for generation
                      (color may differ from the input if it was inferred)
    description     : generated product description from the LoRA model
    similar_products: top-k retrieved products from the catalogue
    plan_trace      : ordered list of PlanStep records
    total_ms        : total wall-clock time for the full agent run
    """
    name: str
    color: str
    description: str
    similar_products: List[Dict[str, Any]] = field(default_factory=list)
    plan_trace: List[PlanStep] = field(default_factory=list)
    total_ms: float = 0.0

    def print_summary(self, max_desc_len: int = 300) -> None:
        """Pretty-print the result to stdout."""
        sep = "=" * 57
        print(f"\n{sep}")
        print("  PLANNER AGENT RESULT")
        print(sep)
        print(f"  Product : {self.name}")
        print(f"  Color   : {self.color or '(none)'}")
        print(f"\n  Generated Description:")
        wrapped = textwrap.fill(self.description[:max_desc_len], width=54,
                                initial_indent="    ", subsequent_indent="    ")
        print(wrapped)
        if len(self.description) > max_desc_len:
            print("    [...]")

        if self.similar_products:
            print(f"\n  Top {len(self.similar_products)} Similar Products (from catalogue):")
            for i, p in enumerate(self.similar_products, 1):
                score = p.get("similarity_score", 0)
                name  = p.get("name_clean", "?")
                color = p.get("colors_extracted", "")
                print(f"    {i}. [{score:.3f}] {name}  |  {color}")

        print(f"\n  Plan Trace:")
        for step in self.plan_trace:
            print(f"    {step}")

        print(f"\n  Total time: {self.total_ms:.0f} ms")
        print(f"{sep}\n")


# ─────────────────────────────────────────────
# Planner Agent
# ─────────────────────────────────────────────

class PlannerAgent:
    """
    Orchestrates the full Retrieve → Generate pipeline for product descriptions.

    Parameters
    ----------
    top_k : int
        Number of similar products to retrieve per query (default 3).
    adapter_path : str
        Path to the LoRA adapter folder (default "lora_model").
    base_model : str
        Base GPT-2 model name (default "gpt2").
    max_new_tokens : int
        Generation length cap (default 120).

    The generator (heavy) is loaded once at construction time.
    The retriever (lighter) is also loaded once; if the FAISS index does not
    exist it is skipped with a warning rather than raising an error.
    """

    def __init__(
        self,
        top_k: int = 3,
        adapter_path: str = "lora_model",
        base_model: str = "gpt2",
        max_new_tokens: int = 120,
        no_retrieval: bool = False,
    ) -> None:
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

        # -- load generator (always required) ----------------------------------
        print("[planner] Loading generator ...")
        self._generator = ProductDescriptionGenerator(
            adapter_path=adapter_path,
            base_model=base_model,
        )

        # -- load retriever (optional) ----------------------------------------
        self._retriever: Optional[ProductRetriever] = None
        if no_retrieval:
            print("[planner] Retrieval disabled via --no-retrieval flag.")
        elif not _retriever_available:
            print("[planner] WARNING: retrieval module not importable — "
                  "running in generation-only mode.")
        else:
            try:
                print("[planner] Loading retriever ...")
                self._retriever = ProductRetriever()
                print("[planner] Retriever ready.")
            except FileNotFoundError as exc:
                print(f"[planner] WARNING: {exc}")
                print("[planner] Run  python scripts/build_index.py  to enable "
                      "retrieval. Continuing in generation-only mode.")

        print("[planner] Agent ready.\n")

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _now_ms() -> float:
        return time.perf_counter() * 1000

    # ── plan steps ────────────────────────────────────────────────────────────

    def _step_parse(self, name: str, color: str) -> tuple[str, str, PlanStep]:
        """STEP 1: Validate / normalise the inputs."""
        t0 = self._now_ms()
        name  = name.strip()
        color = color.strip()
        if not name:
            raise ValueError("Product name must not be empty.")
        detail = f"name='{name}'  color='{color or '(empty)'}'"
        step = PlanStep("PARSE", "ok", detail, self._now_ms() - t0)
        return name, color, step

    def _step_retrieve(
        self, name: str, color: str
    ) -> tuple[List[Dict[str, Any]], PlanStep]:
        """STEP 2: Retrieve the most similar products from the catalogue."""
        t0 = self._now_ms()
        if self._retriever is None:
            step = PlanStep("RETRIEVE", "skipped",
                            "FAISS index not loaded — retrieval disabled.",
                            self._now_ms() - t0)
            return [], step

        similar = self._retriever.retrieve_for_prompt(name, color, top_k=self.top_k)
        detail = (
            f"found {len(similar)} products  "
            f"(top score: {similar[0]['similarity_score']:.3f})"
            if similar else "no results"
        )
        step = PlanStep("RETRIEVE", "ok", detail, self._now_ms() - t0)
        return similar, step

    def _step_enrich(
        self, color: str, similar: List[Dict[str, Any]]
    ) -> tuple[str, PlanStep]:
        """STEP 3: Infer color from retrieved products if the user left it blank."""
        t0 = self._now_ms()
        if color or not similar:
            status = "skipped"
            detail = (
                f"color already provided: '{color}'"
                if color else "no retrieved products to infer from"
            )
        else:
            # Take the color from the highest-scoring retrieved product.
            inferred = similar[0].get("colors_extracted", "").strip()
            if inferred:
                color  = inferred
                status = "ok"
                detail = f"inferred color='{color}' from top retrieved product"
            else:
                status = "warn"
                detail = "retrieved product has no color data; continuing without color"

        step = PlanStep("ENRICH", status, detail, self._now_ms() - t0)
        return color, step

    def _step_generate(
        self, name: str, color: str
    ) -> tuple[str, PlanStep]:
        """STEP 4: Run the fine-tuned LoRA GPT-2 model."""
        t0 = self._now_ms()
        description = self._generator.generate(
            name, color, max_new_tokens=self.max_new_tokens
        )
        tokens = len(description.split())
        detail = f"generated {tokens} tokens"
        step = PlanStep("GENERATE", "ok", detail, self._now_ms() - t0)
        return description, step

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, name: str, color: str = "") -> PlanResult:
        """
        Execute the full planning pipeline for one product.

        Parameters
        ----------
        name  : Product name (required).
        color : Product color (optional — will be inferred if blank and
                the retrieval index is available).

        Returns
        -------
        PlanResult with description, similar products, plan trace, and timing.
        """
        t_start = self._now_ms()
        trace: List[PlanStep] = []

        name,  color,   s1 = self._step_parse(name, color)
        trace.append(s1)

        similar,        s2 = self._step_retrieve(name, color)
        trace.append(s2)

        color,          s3 = self._step_enrich(color, similar)
        trace.append(s3)

        description,    s4 = self._step_generate(name, color)
        trace.append(s4)

        return PlanResult(
            name=name,
            color=color,
            description=description,
            similar_products=similar,
            plan_trace=trace,
            total_ms=self._now_ms() - t_start,
        )

    def run_batch(
        self, items: List[tuple[str, str]]
    ) -> List[PlanResult]:
        """
        Run the agent over a list of (name, color) tuples.
        The model stays loaded between calls.
        """
        results = []
        total = len(items)
        for i, (name, color) in enumerate(items, 1):
            print(f"\n[planner] Batch {i}/{total}: '{name}' | '{color or '(no color)'}'" )
            results.append(self.run(name, color))
        return results

    def run_batch_from_csv(self, csv_path: str) -> List[PlanResult]:
        """
        Load (name, color) pairs from a CSV file and run the batch.
        The CSV must have a 'name' column; 'color' is optional.
        """
        import csv
        items = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "name" not in (reader.fieldnames or []):
                raise ValueError(f"CSV must have a 'name' column. Found: {reader.fieldnames}")
            for row in reader:
                name  = row.get("name", "").strip()
                color = row.get("color", "").strip()
                if name:
                    items.append((name, color))
        print(f"[planner] Loaded {len(items)} products from '{csv_path}'")
        return self.run_batch(items)

    def save_batch_results(
        self, results: List[PlanResult], out_path: str = "batch_results.csv"
    ) -> None:
        """Save batch results to a CSV for easy review."""
        import csv
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["name", "color", "description",
                               "top_similar", "total_ms"]
            )
            writer.writeheader()
            for r in results:
                top = r.similar_products[0]["name_clean"] if r.similar_products else ""
                writer.writerow({
                    "name":        r.name,
                    "color":       r.color,
                    "description": r.description,
                    "top_similar": top,
                    "total_ms":    round(r.total_ms),
                })
        print(f"[planner] Results saved to '{out_path}'"  )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _interactive(agent: PlannerAgent) -> None:
    print("\nInteractive mode — type a product name and (optional) color.")
    print("Press Ctrl-C to quit.\n")
    try:
        while True:
            name  = input("Product name  : ").strip()
            if not name:
                continue
            color = input("Color (or Enter to skip): ").strip()
            result = agent.run(name, color)
            result.print_summary()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Planner Agent: retrieve → generate product descriptions."
    )
    parser.add_argument("--name",          type=str, default="",
                        help="Product name.")
    parser.add_argument("--color",         type=str, default="",
                        help="Product color (optional; inferred if blank).")
    parser.add_argument("--top-k",         type=int, default=3,
                        help="Number of similar products to retrieve (default 3).")
    parser.add_argument("--max-new-tokens", type=int, default=120,
                        help="Max tokens to generate (default 120).")
    parser.add_argument("--adapter-path",  type=str, default="lora_model",
                        help="Path to the LoRA adapter folder.")
    parser.add_argument("--interactive",   action="store_true",
                        help="Run an interactive prompt loop.")
    parser.add_argument("--no-retrieval",  action="store_true",
                        help="Skip retrieval (generation-only mode). "
                             "Use this if the sentence-transformers model "
                             "download is slow or the FAISS index isn't built yet.")
    parser.add_argument("--batch-file",    type=str, default="",
                        help="Path to a CSV file with 'name' (required) and "
                             "'color' (optional) columns. Runs all rows through "
                             "the agent and saves results to batch_results.csv.")
    parser.add_argument("--out",           type=str, default="batch_results.csv",
                        help="Output CSV path for batch results (default: batch_results.csv).")
    args = parser.parse_args()

    agent = PlannerAgent(
        top_k=args.top_k,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        no_retrieval=args.no_retrieval,
    )

    if args.interactive:
        _interactive(agent)
        return

    if args.batch_file:
        results = agent.run_batch_from_csv(args.batch_file)
        agent.save_batch_results(results, out_path=args.out)
        print(f"\n[planner] Batch complete: {len(results)} products processed.")
        for r in results:
            print(f"  • {r.name} ({r.color}) → {r.description[:80]}...")
        return

    name = args.name or "Casual Cotton T-Shirt"
    if not args.name:
        print("[planner] No --name given; using demo example.")

    result = agent.run(name, args.color)
    result.print_summary()


if __name__ == "__main__":
    main()
