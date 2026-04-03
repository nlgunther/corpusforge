"""
CorpusForge CLI
================
Entry point for the `cf` command. Pure presentation layer — no business logic.

Each command handler: parse args → call orchestrator → display result with rich.
Business logic lives in ingester.py (and future: searcher.py, clusterer.py, etc.).

Commands:
    cf ingest <path>       Ingest a Markdown file into the corpus.
    cf status              Show corpus statistics.
    cf files               List all ingested files.
    cf file show <id>      Show detail for a single file.
    cf search <query>      Semantic search across all chunks.
"""

import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .db import CorpusDB
from .ingester import ingest_file, IngestResult

# Force standard output to UTF-8 so Windows file redirection (>) doesn't crash on special characters
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

console = Console()


# ═══════════════════════════════════════════════════════════════════════
# Command handlers — each is ≤ ~15 lines of pure presentation logic
# ═══════════════════════════════════════════════════════════════════════

def cmd_ingest(args: argparse.Namespace) -> None:
    """cf ingest <filepath>"""
    console.print(f"\n[bold blue]Ingesting:[/bold blue] {Path(args.filepath).name}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running ingest pipeline...", total=None)

            result: IngestResult = ingest_file(args.filepath, summarize=args.summarize)

            progress.update(task, completed=100, description="[green]Done")

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except RuntimeError as e:
        # Catches embedding model mismatch and schema version errors.
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    if result.action == "unchanged":
        console.print("[yellow]File unchanged. Skipping ingest.[/yellow]\n")
        return

    action_label = "[green]Inserted[/green]" if result.action == "inserted" else "[blue]Updated[/blue]"
    console.print(f"\n[bold]{action_label}[/bold] '{result.filename}' — {result.chunk_count} chunks")

    if result.auto_summary:
        console.print("\n[bold]LLM Auto-Summary:[/bold]")
        console.print(f"> {result.auto_summary.replace(chr(10), chr(10) + '> ')}\n")


def cmd_status(args: argparse.Namespace) -> None:
    """cf status — corpus overview dashboard."""
    db = CorpusDB()
    stats = db.get_corpus_stats()

    # Two-column table: right-aligned labels, left-aligned values.
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value",  style="magenta", justify="left")

    table.add_row("Total Files Ingested:",    str(stats["file_count"]))
    table.add_row("Total Semantic Chunks:",   str(stats["chunk_count"]))
    table.add_row("Embedded Vectors:",        str(stats["embedded_chunk_count"]))
    table.add_row("Active Topics:",           str(stats["topic_count"]))
    table.add_row("Pending Merge Proposals:", str(stats["pending_cross_refs"]))
    table.add_row("Feedback Entries:",        str(stats["total_feedback"]))

    console.print()
    console.print(Panel(table, title="[bold blue]CorpusForge Status[/bold blue]",
                        expand=False, border_style="blue"))

    if stats["corpus_summary"]:
        console.print("\n[bold]High-Level Corpus Summary:[/bold]")
        console.print(f"> {stats['corpus_summary']}")
    console.print()

def cmd_file_delete(args: argparse.Namespace) -> None:
    """cf file delete <id>"""
    db = CorpusDB()
    f = db.get_file(args.file_id)
    if f is None:
        console.print(f"[bold red]Error:[/bold red] No file with ID {args.file_id}.")
        sys.exit(1)
    console.print(f"Delete '[cyan]{f['filename']}[/cyan]' and all its chunks? [bold](y/N)[/bold] ", end="")
    if input().strip().lower() != "y":
        console.print("Aborted.")
        return
    db.delete_file(args.file_id)
    console.print(f"[green]Deleted.[/green]")

def cmd_files(args: argparse.Namespace) -> None:
    """cf files — list all ingested files."""
    db = CorpusDB()
    files = db.get_all_files()

    if not files:
        console.print("[yellow]No files ingested yet. Run `cf ingest <path>` to start.[/yellow]")
        return

    table = Table(title=f"{len(files)} File(s) in Corpus")
    table.add_column("ID",      style="dim",  no_wrap=True)
    table.add_column("Title",   style="bold")
    table.add_column("File",    style="cyan")
    table.add_column("Format",  no_wrap=True)
    table.add_column("Updated", style="dim",  no_wrap=True)

    for f in files:
        table.add_row(
            str(f["id"]),
            f["title"] or "[dim]—[/dim]",
            f["filename"],
            f["format"],
            f["updated_at"][:10],  # date portion only
        )

    console.print()
    console.print(table)
    console.print()


def cmd_file_show(args: argparse.Namespace) -> None:
    """cf file show <id> — show full detail for a single file."""
    db = CorpusDB()
    f = db.get_file(args.file_id)

    if f is None:
        console.print(f"[bold red]Error:[/bold red] No file with ID {args.file_id}.")
        sys.exit(1)

    chunks = db.get_chunks_for_file(f["id"])

    console.print(f"\n[bold]{f['title'] or f['filename']}[/bold]  [dim](id={f['id']})[/dim]")
    console.print(f"Path:     {f['filepath']}")
    console.print(f"Format:   {f['format']}   Hash: [dim]{f['file_hash'][:12]}…[/dim]")
    console.print(f"Ingested: {f['ingested_at'][:10]}   Updated: {f['updated_at'][:10]}")
    console.print(f"Chunks:   {len(chunks)}")

    if f["auto_summary"]:
        console.print("\n[bold]Auto-Summary:[/bold]")
        console.print(f["auto_summary"])

    if f["user_summary"]:
        console.print("\n[bold]Your Notes:[/bold]")
        console.print(f["user_summary"])

    console.print()


def cmd_search(args: argparse.Namespace) -> None:
    """
    cf search <query> — semantic similarity search across all chunks.

    Embeds the query string, then ranks all stored chunk embeddings by
    cosine similarity. Returns top-k results.
    """
    from .embedder import CorpusEmbedder
    import numpy as np

    db = CorpusDB()
    embedder = CorpusEmbedder()

    # Validate embedding model consistency before any similarity work.
    try:
        db.assert_embedding_model(embedder.model_name)
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    all_embeddings = db.get_all_embeddings()  # [(chunk_id, np.ndarray), ...]
    if not all_embeddings:
        console.print("[yellow]No embeddings found. Ingest some files first.[/yellow]")
        return

    # Embed the query using the same model and prefix convention as ingest.
    query_vec = embedder.model.encode([args.query], convert_to_numpy=True)[0].astype(np.float32)

    # Score all chunks and take the top-k.
    k = args.top_k
    scored = [
        (chunk_id, CorpusEmbedder.cosine_similarity(query_vec, emb))
        for chunk_id, emb in all_embeddings
    ]
    top_results = sorted(scored, key=lambda x: x[1], reverse=True)[:k]

    console.print(f"\n[bold]Top {k} results for:[/bold] \"{args.query}\"\n")

    for rank, (chunk_id, score) in enumerate(top_results, 1):
        chunk = db.get_chunk(chunk_id)
        file  = db.get_file(chunk["file_id"])
        preview = chunk["content"][:200].replace("\n", " ")
        if len(chunk["content"]) > 200:
            preview += "…"

        console.print(
            f"[bold]{rank}.[/bold] [cyan]{file['filename']}[/cyan]  "
            f"[dim]{chunk['heading_path']}[/dim]  "
            f"[green]{score:.3f}[/green]"
        )
        console.print(f"   {preview}\n")

def cmd_cluster(args: argparse.Namespace) -> None:
    """cf cluster — run HDBSCAN to group chunks into topics."""
    from .topic_engine import TopicEngine
    
    console.print("\n[bold blue]Clustering Corpus...[/bold blue]")
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        # Step 1: The Math
        task_math = progress.add_task("Running HDBSCAN...", total=None)
        engine = TopicEngine(min_cluster_size=3)
        topic_count = engine.cluster_corpus()
        progress.update(task_math, completed=100, description="[green]Vector space clustered[/green]")
        
        # Step 2: The LLM Naming (Opt-In based on CLI flag)
        if args.name_topics:
            task_llm = progress.add_task("Generating topic names with LLM...", total=None)
            engine.name_topics()
            progress.update(task_llm, completed=100, description="[green]Topics named[/green]")
        
    console.print(f"\n[bold green]✓ Success![/bold green] Discovered [bold]{topic_count}[/bold] distinct topics.")
    
    # Context-aware next steps
    if args.name_topics:
        console.print("Run [cyan]cf topics[/cyan] to see your newly generated knowledge graph.\n")
    else:
        console.print("[dim yellow]Run `cf cluster --name-topics` to automatically generate names for these topics.[/dim yellow]")
        console.print("Run [cyan]cf status[/cyan] to see the updated topic count.\n")

def cmd_topics(args: argparse.Namespace) -> None:
    """cf topics — list all discovered topics."""
    db = CorpusDB()
    topics = db.get_all_topics()

    if not topics:
        console.print("[yellow]No topics found. Run `cf cluster` to discover some.[/yellow]")
        return

    # Build a clean, readable table
    table = Table(title=f"{len(topics)} Discovered Topic(s)", show_lines=True)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Topic Name", style="bold cyan")
    table.add_column("Description")

    for t in topics:
        table.add_row(
            str(t["id"]),
            t["name"],
            t["description"] or "[dim]Pending LLM analysis...[/dim]"
        )

    console.print()
    console.print(table)
    console.print()

def cmd_topic_show(args: argparse.Namespace) -> None:
    """cf topic show <id> — dump all text chunks for a specific topic."""
    db = CorpusDB()
    
    # Optional: If you want to show the topic name at the top, we can fetch all topics to find the name
    topics = {t["id"]: t for t in db.get_all_topics()}
    topic = topics.get(args.topic_id)
    
    if not topic:
        console.print(f"[bold red]Error:[/bold red] No topic found with ID {args.topic_id}.")
        sys.exit(1)

    chunks = db.get_chunks_for_topic(args.topic_id)
    
    if not chunks:
        console.print(f"[yellow]Topic '{topic['name']}' has no chunks assigned.[/yellow]")
        return

    console.print(f"\n[bold blue]Topic {args.topic_id}: {topic['name']}[/bold blue]")
    console.print(f"[dim]{topic['description']}[/dim]\n")
    console.print(f"Contains [bold]{len(chunks)}[/bold] semantic chunks:\n")

    # Loop through and print each chunk with its source file and similarity score
    for idx, c in enumerate(chunks, 1):
        preview = c["content"].strip()
        console.print(f"[bold cyan]--- {idx}. {c['filename']} ---[/bold cyan] [dim](Relevance: {c['similarity']:.3f})[/dim]")
        
        # If the parser caught a Markdown header, display it
        if c["heading_path"] and c["heading_path"] != "Root":
            console.print(f"[dim]Section: {c['heading_path']}[/dim]")
            
        console.print(f"{preview}\n")

def cmd_db_optimize(args: argparse.Namespace) -> None:
    """cf db optimize — reclaim space and update query planner statistics."""
    db = CorpusDB()
    console.print("[blue]Optimizing database...[/blue]")
    # VACUUM and ANALYZE cannot run inside a transaction, so we open a
    # raw connection and close it explicitly rather than using db.transaction().
    conn = db._make_connection()
    try:
        conn.execute("VACUUM")   # reclaim pages freed by DELETE operations
        conn.execute("ANALYZE")  # update query planner statistics
    finally:
        conn.close()
    console.print("[bold green]✓ Database optimized![/bold green] (Reclaimed empty pages and updated query planner)")

def cmd_file_export(args: argparse.Namespace) -> None:
    from .compiler import LocalCompiler
    result = LocalCompiler().export_tagged_document(args.file_id)
    if not result:
        console.print(f"[bold red]Error:[/bold red] Could not export File {args.file_id}.")
    else:
        console.print(f"[bold green]✓ Success![/bold green] Exported {result.chunk_count} chunks to: [cyan]{result.output_path}[/cyan]")

def cmd_topic_outline(args: argparse.Namespace) -> None:
    from .compiler import LocalCompiler
    result = LocalCompiler().generate_topic_outline(args.topic_id)
    if result:
        console.print(f"[bold green]✓ Success![/bold green] Outline saved to: [cyan]{result.output_path}[/cyan]")

def cmd_topic_export_linear(args: argparse.Namespace) -> None:
    from .compiler import LocalCompiler
    result = LocalCompiler().compile_linear_document(args.topic_id)
    if result:
        console.print(f"[bold green]✓ Success![/bold green] Linear document saved to: [cyan]{result.output_path}[/cyan]")

def cmd_topic_compile(args: argparse.Namespace) -> None:
    """cf topic compile <id> — synthesize topic chunks into prose via LLM (max 12 chunks)."""
    from .compiler import LocalCompiler
    from .summarizer import CorpusSummarizer

    console.print(f"\n[bold blue]Compiling Topic {args.topic_id}...[/bold blue]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Weaving narrative with LLM (max 12 chunks)...", total=None)
        result = LocalCompiler().compile_topic_llm(args.topic_id, summarizer=CorpusSummarizer())
        progress.update(task, completed=100, description="[green]Compilation complete[/green]")

    if result:
        console.print(f"\n[bold green]✓ Success![/bold green] Synthesized document saved to: [cyan]{result.output_path}[/cyan]\n")

def cmd_topic_summarize(args: argparse.Namespace) -> None:
    """cf topic summarize <id> — Generate English name/desc for a single topic."""
    from .summarizer import CorpusSummarizer
    db = CorpusDB()

    chunks = db.get_chunks_for_topic(args.topic_id)
    if not chunks:
        console.print(f"[yellow]No chunks found for Topic {args.topic_id}.[/yellow]")
        return

    console.print(f"Analyzing Topic {args.topic_id} with Gemini...")
    summarizer = CorpusSummarizer()

    # Take top 3 chunks (matches Claude's logic in topic_engine to save tokens)
    sample_texts = [c["content"] for c in chunks[:3]]
    metadata = summarizer.name_topic(sample_texts)

    db.update_topic(
        topic_id=args.topic_id,
        name=metadata.get("name", "Unnamed Topic"),
        description=metadata.get("description", "")
    )

    console.print(f"\n[bold green]✓ Topic Updated![/bold green]")
    console.print(f"[bold]New Name:[/bold] {metadata.get('name')}")
    console.print(f"[bold]Description:[/bold] {metadata.get('description')}\n")

# ═══════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cf",
        description="CorpusForge — local document corpus management CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # cf ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a Markdown file into the corpus")
    p_ingest.add_argument("filepath", help="Path to the Markdown file")
    p_ingest.add_argument("--summarize", action="store_true", dest="summarize",
        help="Generate an LLM summary after ingestion (requires GEMINI_API_KEY)")

    # cf status
    sub.add_parser("status", help="Show corpus statistics")

    # cf files
    sub.add_parser("files", help="List all ingested files")

    # cf topics
    sub.add_parser("topics", help="List all discovered semantic topics")

    # cf file ...
    p_file = sub.add_parser("file", help="File operations")
    file_sub = p_file.add_subparsers(dest="file_command", metavar="<subcommand>")
    p_file_show = file_sub.add_parser("show", help="Show detail for a single file")
    p_file_show.add_argument("file_id", type=int, help="File ID (from cf files)")
    p_file_delete = file_sub.add_parser("delete", help="Remove a file and all its data")
    p_file_delete.add_argument("file_id", type=int, help="File ID (from cf files)")
    
    # MISSING LINE RESTORED HERE:
    p_file_export = file_sub.add_parser("export-tagged", help="Reconstruct file with chunk tags")
    p_file_export.add_argument("file_id", type=int, help="File ID to export")

    # cf db ...
    p_db = sub.add_parser("db", help="Database operations")
    db_sub = p_db.add_subparsers(dest="db_command", metavar="<subcommand>")
    db_sub.add_parser("optimize", help="Run VACUUM and ANALYZE to reclaim space")

    # cf search <query>
    p_search = sub.add_parser("search", help="Semantic search across all chunks")
    p_search.add_argument("query", help="Natural language search query")
    p_search.add_argument(
        "--top-k", type=int, default=5, dest="top_k",
        help="Number of results to return (default: 5)"
    )
    
    # cf cluster
    p_cluster = sub.add_parser("cluster", help="Run HDBSCAN to group chunks into semantic topics")
    p_cluster.add_argument("--name-topics", action="store_true", dest="name_topics",
        help="Use the LLM to generate human-readable names for discovered topics")

    # cf topic ...
    p_topic = sub.add_parser("topic", help="Topic operations")
    topic_sub = p_topic.add_subparsers(dest="topic_command", metavar="<subcommand>")
    p_topic_show = topic_sub.add_parser("show", help="Show all text chunks for a specific topic")
    p_topic_show.add_argument("topic_id", type=int, help="Topic ID (from cf topics)")

    # Phase 3 Compiler tools (DUPLICATE REMOVED)
    topic_sub.add_parser("compile", help="Weave chunks into a synthesized Markdown document").add_argument("topic_id", type=int)
    topic_sub.add_parser("outline", help="Generate a hierarchical checklist for a topic").add_argument("topic_id", type=int)
    topic_sub.add_parser("export-linear", help="Reconstruct topic chunks in their original reading order").add_argument("topic_id", type=int)

    topic_sub.add_parser("summarize", help="Generate English name/desc for a single topic").add_argument("topic_id", type=int)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "status": cmd_status,
        "files":  cmd_files,
        "search": cmd_search,
        "cluster": cmd_cluster,  # <--- NEW
        "topics": cmd_topics,  # <--- ADD THIS
    }

    subcommand_dispatch = {
            "file": {
                "show": cmd_file_show, 
                "delete": cmd_file_delete, 
                "export-tagged": cmd_file_export
            },
            "topic": {
                "show": cmd_topic_show, 
                "compile": cmd_topic_compile, 
                "outline": cmd_topic_outline, 
                "export-linear": cmd_topic_export_linear
                "summarize": cmd_topic_summarize  # <--- ADD THIS LINE!
            },
            "db": {
                "optimize": cmd_db_optimize
            }
        }

    if args.command in subcommand_dispatch:
        sub_cmd = getattr(args, f"{args.command}_command")
        if sub_cmd in subcommand_dispatch[args.command]:
            subcommand_dispatch[args.command][sub_cmd](args)
        else:
            parser.parse_args([args.command, "--help"])
    elif args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
