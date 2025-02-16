from dotenv import load_dotenv
import asyncio
import typer
from functools import wraps
from prompt_toolkit import PromptSession
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from deep_research_py.deep_research import deep_research, write_final_report
from deep_research_py.feedback import generate_feedback
from deep_research_py.ai.providers import get_ai_client

from deep_research_py.utils import ModelConfig

load_dotenv()
app = typer.Typer()
session = PromptSession()


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


async def async_prompt(message: str, default: str = "") -> str:
    """Async wrapper for prompt_toolkit."""
    return await session.prompt_async(message)


@app.command()
@coro
async def main(
    concurrency: int = typer.Option(
        default=2, help="Number of concurrent tasks, depending on your API rate limits."
    ),
    service: str = typer.Option(
        default="openai",
        help="Which service to use? [openai|deepseek]",
    ),
    model: str = typer.Option(default="o3-mini", help="Which model to use?"),
    max_followup_questions: int = typer.Option(
        default=5,
        help="Maximum number of follow-up questions to generate.",
    ),
    enable_logging: bool = typer.Option(
        default=False,
        help="Enable logging.",
    ),
    log_path: str = typer.Option(
        default="logs",
        help="Path to save the logs.",
    ),
    log_to_stdout: bool = typer.Option(
        default=False,
        help="Log to stdout.",
    ),
):
    ModelConfig.set_service(service.lower())
    ModelConfig.set_model(model)
    console = ModelConfig.console

    """Initialize the Logger"""
    if enable_logging:
        from deep_research_py.common.logging import initial_logger

        initial_logger(logging_path=log_path, enable_stdout=log_to_stdout)
        console.print(f"[dim]Logging enabled. Logs will be saved to {log_path}[/dim]")

    """Deep Research CLI"""
    console.print(
        Panel.fit(
            "[bold blue]Deep Research Assistant[/bold blue]\n"
            "[dim]An AI-powered research tool[/dim]"
        )
    )

    console.print(f"🛠️ Using [bold green]{service.upper()}[/bold green] service.")

    client = get_ai_client()

    # Get initial inputs with clear formatting
    query = await async_prompt("\n🔍 What would you like to research? ")
    console.print()

    breadth_prompt = "📊 Research breadth (recommended 2-10) [4]: "
    breadth = int((await async_prompt(breadth_prompt)) or "4")
    console.print()

    depth_prompt = "🔍 Research depth (recommended 1-5) [2]: "
    depth = int((await async_prompt(depth_prompt)) or "2")
    console.print()

    # First show progress for research plan
    console.print("\n[yellow]Creating research plan...[/yellow]")
    follow_up_questions = await generate_feedback(
        query, client, model, max_followup_questions
    )

    if len(follow_up_questions) != 0:
        # Then collect answers separately from progress display
        console.print("\n[bold yellow]Follow-up Questions:[/bold yellow]")
        answers = []
        for i, question in enumerate(follow_up_questions, 1):
            console.print(f"\n[bold blue]Q{i}:[/bold blue] {question}")
            answer = await async_prompt("➤ Your answer: ")
            answers.append(answer)
            console.print()

    else:
        console.print("\n[bold green]No follow-up questions needed![/bold green]")
        answers = []

    # Combine information
    combined_query = f"""
    Initial Query: {query}
    Follow-up Questions and Answers:
    {chr(10).join(f"Q: {q} A: {a}" for q, a in zip(follow_up_questions, answers))}
    """

    # Now use Progress for the research phase
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Do research
        task = progress.add_task(
            "[yellow]Researching your topic...[/yellow]", total=None
        )
        research_results = await deep_research(
            query=combined_query,
            breadth=breadth,
            depth=depth,
            concurrency=concurrency,
            client=client,
            model=model,
        )
        progress.remove_task(task)

        # Show learnings
        console.print("\n[yellow]Learnings:[/yellow]")
        for learning in research_results["learnings"]:
            rprint(f"• {learning}")

        # Generate report
        task = progress.add_task("Writing final report...", total=None)
        report = await write_final_report(
            prompt=combined_query,
            learnings=research_results["learnings"],
            visited_urls=research_results["visited_urls"],
            client=client,
            model=model,
        )
        progress.remove_task(task)

        # Show results
        console.print("\n[bold green]Research Complete![/bold green]")
        console.print("\n[yellow]Final Report:[/yellow]")
        console.print(Panel(report, title="Research Report"))

        # Show sources
        console.print("\n[yellow]Sources:[/yellow]")
        for url in research_results["visited_urls"]:
            rprint(f"• {url}")

        # Save report
        with open("output.md", "w") as f:
            f.write(report)
        console.print("\n[dim]Report has been saved to output.md[/dim]")


def run():
    """Synchronous entry point for the CLI tool."""
    asyncio.run(app())


if __name__ == "__main__":
    asyncio.run(app())
