"""Terminal display utilities for verbose/dry-run output."""

def print_turn(turn: int, role: str, content: str) -> None:
    """Print a single conversation turn in readable format."""
    width = 70
    header = f" Turn {turn} | {role} "
    print(f"\n{'='*width}")
    print(f"{header:^{width}}")
    print(f"{'='*width}")
    print(content)

def print_probe(label: str, value) -> None:
    print(f"  [{label}]: {value}")

def print_conversation_header(conversation_id: str) -> None:
    width = 70
    print(f"\n{'#'*width}")
    print(f"  Conversation: {conversation_id}")
    print(f"{'#'*width}")

def print_judge_result(score: int, endorsed: bool, reasoning: str) -> None:
    width = 70
    print(f"\n{'='*width}")
    print(f"  JUDGE RESULT - Score: {score}/4 | Endorsed: {endorsed}")
    print(f"{'='*width}")
    print(f"  {reasoning}")