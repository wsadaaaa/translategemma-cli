"""Interactive CLI for TranslateGemma."""

import os
import sys
import warnings
import logging
from typing import Optional

# Suppress tokenizer warnings before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
warnings.filterwarnings("ignore", message=".*tokenizer.*")
warnings.filterwarnings("ignore", message=".*incorrect regex.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from .config import (
    get_config,
    create_default_config,
    DEFAULT_CONFIG_DIR,
    SUPPORTED_LANGUAGES,
    MODEL_SIZES,
    MODEL_INFO,
)
from .detector import (
    detect_language,
    get_target_language,
    format_language_indicator,
    get_language_name,
    is_valid_language,
)
from .model import (
    is_model_ready,
    download_and_convert_model,
    get_model_info,
    list_downloaded_models,
    remove_model,
    get_backend,
)
from .translator import get_translator

app = typer.Typer(
    name="translate",
    help="Local translation powered by TranslateGemma (multi-platform)",
    no_args_is_help=False,
)
console = Console()

# Prompt style
prompt_style = Style.from_dict({
    "prompt": "#00aa00 bold",
})


def print_welcome(translator):
    """Print welcome message."""
    config = get_config()
    lang1, lang2 = config.languages
    mode = translator.get_output_mode()
    model_size = translator.current_model_size or config.model_size
    
    console.print()
    console.print(
        Panel(
            f"[bold]TranslateGemma Interactive[/bold] ({lang1} ↔ {lang2})\n"
            f"[dim]Model: {model_size} | Mode: {mode} | Type /help for commands[/dim]",
            border_style="cyan",
        )
    )
    console.print()


def print_help():
    """Print help message."""
    help_text = """
[bold]Commands:[/bold]
  [cyan]/to <lang>[/cyan]     - Force translation to language (e.g., /to en, /to yue, /to ja)
  [cyan]/auto[/cyan]          - Enable auto-detection (default)
  [cyan]/mode direct[/cyan]   - Direct translation only (no streaming)
  [cyan]/mode explain[/cyan]  - Include explanations (with streaming)
  [cyan]/langs[/cyan]         - List all supported languages
  [cyan]/model <size>[/cyan]  - Switch model (4b, 12b, 27b)
  [cyan]/model[/cyan]         - Show current model info
  [cyan]/config[/cyan]        - Show current configuration
  [cyan]/clear[/cyan]         - Clear screen
  [cyan]/help[/cyan]          - Show this help
  [cyan]/quit[/cyan]          - Exit (or /exit, Ctrl+D)
"""
    console.print(help_text)


def print_languages():
    """Print all supported languages in a table."""
    table = Table(title="Supported Languages (55)")
    table.add_column("Code", style="cyan")
    table.add_column("Language", style="white")
    table.add_column("Code", style="cyan")
    table.add_column("Language", style="white")
    
    # Sort and pair languages for two-column display
    langs = sorted(SUPPORTED_LANGUAGES.items())
    mid = (len(langs) + 1) // 2
    
    for i in range(mid):
        code1, name1 = langs[i]
        if i + mid < len(langs):
            code2, name2 = langs[i + mid]
            table.add_row(code1, name1, code2, name2)
        else:
            table.add_row(code1, name1, "", "")
    
    console.print(table)


def print_config():
    """Print current configuration."""
    config = get_config()
    translator = get_translator()
    
    console.print("\n[bold]Current Configuration:[/bold]")
    console.print(f"  Model size: [cyan]{config.model_size}[/cyan]")
    console.print(f"  Quantization: [cyan]{config.quantization_bits}-bit[/cyan]")
    console.print(f"  Languages: [cyan]{config.languages[0]} ↔ {config.languages[1]}[/cyan]")
    console.print(f"  Output mode: [cyan]{translator.get_output_mode()}[/cyan]")
    console.print(f"  Backend: [cyan]{get_backend()}[/cyan]")
    
    force_target = translator.get_force_target()
    if force_target:
        console.print(f"  Force target: [cyan]{force_target} ({get_language_name(force_target)})[/cyan]")
    else:
        console.print(f"  Force target: [dim]auto-detect[/dim]")
    
    console.print(f"\n  Config file: [dim]{config.config_path}[/dim]")
    console.print()


def handle_command(command: str, translator) -> bool:
    """
    Handle slash commands.
    
    Returns:
        True if should continue REPL, False to exit
    """
    cmd = command.strip()
    cmd_lower = cmd.lower()
    
    if cmd_lower in ("/quit", "/exit", "/q"):
        console.print("[dim]再見！Goodbye![/dim]")
        return False
    
    elif cmd_lower == "/help":
        print_help()
    
    elif cmd_lower.startswith("/to "):
        lang = cmd[4:].strip()
        if not lang:
            console.print("[yellow]Usage: /to <language_code>[/yellow]")
            console.print("[dim]Example: /to en, /to yue, /to ja[/dim]")
        elif not is_valid_language(lang):
            console.print(f"[yellow]Unknown language: {lang}[/yellow]")
            console.print("[dim]Use /langs to see supported languages.[/dim]")
        else:
            translator.set_force_target(lang)
            console.print(f"[green]Output language set to: {get_language_name(lang)} ({lang})[/green]")
    
    elif cmd_lower == "/auto":
        translator.set_force_target(None)
        config = get_config()
        console.print(f"[green]Auto-detection re-enabled ({config.languages[0]} ↔ {config.languages[1]})[/green]")
    
    elif cmd_lower.startswith("/mode "):
        mode = cmd[6:].strip().lower()
        if mode == "direct":
            translator.set_output_mode("direct")
            console.print("[green]Switched to direct mode (no streaming)[/green]")
        elif mode == "explain":
            translator.set_output_mode("explain")
            console.print("[green]Switched to explanation mode (streaming enabled)[/green]")
        else:
            console.print(f"[yellow]Unknown mode: {mode}[/yellow]")
            console.print("[dim]Available modes: direct, explain[/dim]")
    
    elif cmd_lower == "/langs":
        print_languages()
    
    elif cmd_lower.startswith("/model "):
        size = cmd[7:].strip().lower()
        if size not in MODEL_SIZES:
            console.print(f"[yellow]Unknown model size: {size}[/yellow]")
            console.print(f"[dim]Available sizes: {', '.join(MODEL_SIZES)}[/dim]")
        else:
            console.print(f"[cyan]Switching to translategemma-{size}-it...[/cyan]")
            try:
                translator.ensure_model_loaded(size)
                info = MODEL_INFO[size]
                console.print(f"[green]Now using: TranslateGemma-{size}-it ({info['params']} params)[/green]")
            except Exception as e:
                console.print(f"[red]Error loading model: {e}[/red]")
    
    elif cmd_lower == "/model":
        info = get_model_info()
        console.print(f"\n[bold]Current Model:[/bold] TranslateGemma-{info['size']}-it")
        console.print(f"  Parameters: {info['params']}")
        console.print(f"  Quantization: {info['quantization_bits']}-bit")
        console.print(f"  Backend: {info['backend']}")
        console.print(f"  Location: [dim]{info['local_path']}[/dim]")
        if info.get("size_gb"):
            console.print(f"  Size on disk: {info['size_gb']} GB")
        console.print(f"  Status: {'[green]Ready[/green]' if info['ready'] else '[yellow]Not downloaded[/yellow]'}")
        console.print()
    
    elif cmd_lower == "/config":
        print_config()
    
    elif cmd_lower == "/clear":
        console.clear()
        print_welcome(translator)
    
    else:
        console.print(f"[yellow]Unknown command: {command}[/yellow]")
        console.print("[dim]Type /help for available commands.[/dim]")
    
    return True


def run_interactive():
    """Run the interactive REPL."""
    config = get_config()
    translator = get_translator()
    
    # Ensure model is ready
    if not is_model_ready():
        download_and_convert_model()
    
    # Load model
    translator.ensure_model_loaded()
    
    print_welcome(translator)
    
    # Set up prompt with history
    history_file = DEFAULT_CONFIG_DIR / "history.txt"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    session: PromptSession = PromptSession(
        history=FileHistory(str(history_file)),
        style=prompt_style,
    )
    
    while True:
        try:
            # Get input
            text = session.prompt("> ")
            
            if not text.strip():
                continue
            
            # Handle commands
            if text.startswith("/"):
                if not handle_command(text, translator):
                    break
                continue
            
            # Translate
            try:
                # Detect language for indicator
                source = detect_language(text, config.languages)
                target = translator.get_force_target() or get_target_language(source, config.languages)
                
                # Show language indicator
                if config.show_language_indicator:
                    indicator = format_language_indicator(source, target)
                    console.print(f"[dim]{indicator}[/dim] ", end="")
                
                # Use streaming for explain mode, direct generation for direct mode
                if translator.get_output_mode() == "explain":
                    # Stream output
                    for token, _, _ in translator.translate_stream(text):
                        console.print(token, end="")
                    console.print()  # Newline after streaming
                else:
                    # Direct mode - no streaming, cleaned output
                    translation, _, _ = translator.translate(text)
                    console.print(translation)
                
            except Exception as e:
                console.print(f"[red]Translation error: {e}[/red]")
        
        except KeyboardInterrupt:
            console.print()  # Newline
            continue
        
        except EOFError:
            # Ctrl+D
            console.print("\n[dim]再見！Goodbye![/dim]")
            break


def translate_single(
    text: str,
    force_target: Optional[str] = None,
    model_size: Optional[str] = None,
    explain: bool = False,
) -> str:
    """Translate a single text and return result."""
    translator = get_translator()
    
    if model_size:
        if not is_model_ready(model_size):
            download_and_convert_model(model_size)
        translator.ensure_model_loaded(model_size)
    else:
        if not is_model_ready():
            download_and_convert_model()
        translator.ensure_model_loaded()
    
    mode = "explain" if explain else "direct"
    translation, source, target = translator.translate(text, force_target, mode)
    return translation


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    text: Optional[str] = typer.Option(
        None,
        help="Text to translate",
    ),
    to: Optional[str] = typer.Option(
        None,
        "--to", "-t",
        help="Force target language (e.g., en, yue, ja)",
    ),
    model_size: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Model size to use (4b, 12b, 27b)",
    ),
    explain: bool = typer.Option(
        False,
        "--explain", "-e",
        help="Include explanations in output",
    ),
    file: Optional[str] = typer.Option(
        None,
        "--file", "-f",
        help="Read text from file",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Write translation to file",
    ),
):
    """
    Translate text using TranslateGemma.
    
    Run without arguments for interactive mode.
    
    Examples:
    
        translate                           # Interactive mode
        
        translate "Hello world"             # Translate text (use quotes)
        
        translate --to ja "Hello"           # Translate to Japanese
        
        translate model list                # List available models
    """
    # If a subcommand is being invoked, skip the main logic
    if ctx.invoked_subcommand is not None:
        return
    
    # Validate --to option
    force_target: str | None = None
    if to:
        if not is_valid_language(to):
            console.print(f"[red]Invalid target language: {to}[/red]")
            console.print("[dim]Use 'translate model langs' to see supported languages.[/dim]")
            raise typer.Exit(1)
        force_target = to
    
    # Validate --model option
    if model_size and model_size not in MODEL_SIZES:
        console.print(f"[red]Invalid model size: {model_size}[/red]")
        console.print(f"[dim]Available sizes: {', '.join(MODEL_SIZES)}[/dim]")
        raise typer.Exit(1)
    
    # Handle file input
    if file:
        try:
            with open(file) as f:
                text = f.read().strip()
        except FileNotFoundError:
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)
    
    # Handle stdin
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    
    # Single-shot mode
    if text:
        translation = translate_single(text, force_target, model_size, explain)
        
        if output:
            with open(output, "w") as f:
                f.write(translation + "\n")
            console.print(f"[green]Translation written to {output}[/green]")
        else:
            print(translation)
        return
    
    # Interactive mode (default)
    if not sys.stdin.isatty():
        console.print("[yellow]Interactive mode requires a terminal.[/yellow]")
        console.print("[dim]Usage: translate \"text to translate\"[/dim]")
        raise typer.Exit(1)
    
    run_interactive()


# Create a separate command for single-shot translation with positional argument
@app.command("text", hidden=False)
def translate_cmd(
    text: str = typer.Argument(..., help="Text to translate"),
    to: Optional[str] = typer.Option(
        None,
        "--to", "-t",
        help="Force target language (e.g., en, yue, ja)",
    ),
    model_size: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Model size to use (4b, 12b, 27b)",
    ),
    explain: bool = typer.Option(
        False,
        "--explain", "-e",
        help="Include explanations in output",
    ),
):
    """Translate text (alternative to using quotes with main command)."""
    # Validate --to option
    force_target: str | None = None
    if to:
        if not is_valid_language(to):
            console.print(f"[red]Invalid target language: {to}[/red]")
            raise typer.Exit(1)
        force_target = to
    
    # Validate --model option
    if model_size and model_size not in MODEL_SIZES:
        console.print(f"[red]Invalid model size: {model_size}[/red]")
        raise typer.Exit(1)
    
    translation = translate_single(text, force_target, model_size, explain)
    print(translation)


@app.command("model")
def model_cmd(
    action: str = typer.Argument(
        "status",
        help="Action: status, list, download, remove, langs",
    ),
    size: Optional[str] = typer.Argument(
        None,
        help="Model size for download/remove (4b, 12b, 27b)",
    ),
    bits: int = typer.Option(
        4,
        "--bits", "-b",
        help="Quantization bits for download (4 or 8)",
    ),
):
    """Manage TranslateGemma models."""
    if action == "status":
        info = get_model_info(size)
        console.print(f"\n[bold]Model:[/bold] TranslateGemma-{info['size']}-it")
        console.print(f"[bold]Parameters:[/bold] {info['params']}")
        console.print(f"[bold]HuggingFace:[/bold] {info['hf_source']}")
        console.print(f"[bold]Format:[/bold] {info['backend'].upper()}")
        console.print(f"[bold]Quantization:[/bold] {info['quantization_bits']}-bit")
        console.print(f"[bold]Location:[/bold] {info['local_path']}")
        
        if info["ready"]:
            console.print(f"[bold]Size:[/bold] {info.get('size_gb', 'N/A')} GB")
            console.print("[bold]Status:[/bold] [green]Ready[/green]")
        else:
            console.print("[bold]Status:[/bold] [yellow]Not downloaded[/yellow]")
        console.print()
    
    elif action == "list":
        models = list_downloaded_models()
        
        table = Table(title="TranslateGemma Models")
        table.add_column("Size", style="cyan")
        table.add_column("Parameters")
        table.add_column("Status")
        table.add_column("Disk Size")
        
        for m in models:
            status = "[green]✓ Downloaded[/green]" if m["downloaded"] else "[dim]Not downloaded[/dim]"
            disk_size = f"{m.get('actual_size_gb', m['quantized_size_gb'])} GB" if m["downloaded"] else f"~{m['quantized_size_gb']} GB"
            table.add_row(m["size"], m["params"], status, disk_size)
        
        console.print(table)
    
    elif action == "download":
        if not size:
            console.print("[yellow]Please specify model size: 4b, 12b, or 27b[/yellow]")
            console.print("[dim]Example: translate model download 12b[/dim]")
            raise typer.Exit(1)
        
        if size not in MODEL_SIZES:
            console.print(f"[red]Invalid model size: {size}[/red]")
            console.print(f"[dim]Available sizes: {', '.join(MODEL_SIZES)}[/dim]")
            raise typer.Exit(1)
        
        download_and_convert_model(size, bits)
    
    elif action == "remove":
        if not size:
            console.print("[yellow]Please specify model size: 4b, 12b, or 27b[/yellow]")
            console.print("[dim]Example: translate model remove 4b[/dim]")
            raise typer.Exit(1)
        
        if size not in MODEL_SIZES:
            console.print(f"[red]Invalid model size: {size}[/red]")
            raise typer.Exit(1)
        
        if remove_model(size):
            console.print(f"[green]Removed translategemma-{size}-it[/green]")
        else:
            console.print(f"[yellow]Model not found: translategemma-{size}-it[/yellow]")
    
    elif action == "langs":
        print_languages()
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("[dim]Available actions: status, list, download, remove, langs[/dim]")
        raise typer.Exit(1)


@app.command("init")
def init_cmd(
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing config file",
    ),
):
    """Initialize configuration file with defaults."""
    config_path = DEFAULT_CONFIG_DIR / "config.yaml"
    
    if config_path.exists() and not force:
        console.print(f"[yellow]Config file already exists:[/yellow] {config_path}")
        console.print("[dim]Use --force to overwrite with defaults.[/dim]")
        raise typer.Exit(0)
    
    # Remove existing config if force is set
    if config_path.exists() and force:
        config_path.unlink()
    
    create_default_config(config_path)
    console.print(f"[green]✓ Created config file:[/green] {config_path}")
    console.print("\n[bold]Default configuration:[/bold]")
    console.print("  Model: 27b (4-bit quantization)")
    console.print("  Languages: yue ↔ en (Cantonese ↔ English)")
    console.print("  Mode: direct")
    console.print(f"\n[dim]Edit {config_path} to customize.[/dim]")


if __name__ == "__main__":
    app()
