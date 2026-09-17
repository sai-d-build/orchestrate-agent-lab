import os
import sys

from core.llm.config import get_model_config, get_api_key
from core.llm.providers.openrouter import OpenRouterProvider


def main() -> None:
    config = get_model_config()

    print(f"Profile:  {config.profile}")
    print(f"Provider: {config.provider}")
    print(f"Model:    {config.model}")

    api_key = get_api_key(config)

    provider = OpenRouterProvider(
        api_key=api_key,
        model=config.model,
    )

    prompt = """
You are analyzing a small transaction dataset.

Transactions:

T001 | customer=C101 | product=API | quantity=3 | unit_price=120 | status=completed
T002 | customer=C102 | product=DB  | quantity=2 | unit_price=250 | status=completed
T003 | customer=C101 | product=API | quantity=1 | unit_price=120 | status=cancelled
T004 | customer=C103 | product=ML  | quantity=4 | unit_price=175 | status=completed
T005 | customer=C102 | product=DB  | quantity=3 | unit_price=250 | status=completed
T006 | customer=C101 | product=ML  | quantity=2 | unit_price=175 | status=completed

Tasks:

1. Ignore cancelled transactions.
2. Calculate the total revenue from completed transactions.
3. Calculate revenue by customer.
4. Identify the customer with the highest revenue.
5. Identify the highest-revenue product.
6. Explain your calculations briefly.
7. Check your arithmetic before answering.

Return the answer using exactly this structure:

Total revenue: <number>

Revenue by customer:
- C101: <number>
- C102: <number>
- C103: <number>

Highest-revenue customer: <customer> (<number>)

Highest-revenue product: <product> (<number>)

Calculation check: <brief explanation>
"""

    print(f"\nPrompt: {prompt}")

    try:
        response = provider.generate(prompt)
    except Exception as exc:
        print(f"\nAPI request failed: {exc}")
        sys.exit(1)

    print(f"\nAnswer:\n{response.content}")


if __name__ == "__main__":
    main()
