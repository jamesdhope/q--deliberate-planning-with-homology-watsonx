## q--deliberate-planning-watsonx

Implementation of Q*: Improving Multi-step Reasoning for LLMs with Deliberate Planning.

Source: https://arxiv.org/pdf/2406.14283v1

### envars

Export the following envars to your environment:

WATSONX_APIKEY=XXXXXX
PROJECT_ID=XXXXXX

### Implementation Details & Adaptions

* Trajectories are completed by an expert model with a terminal state that is determined by the expert.
* h(s) or the Q-value is the average of the log_probs for the generated sequence
* The aggregated utility h(s) is the aggregated Q-value or log_probs for the path to that state
* The algorithm terminates when the open_list is empty or if the specified number of states has been visited
* The question / task, the number of states that can be visited, the semantic similarity score for states to be considered the same (visited), the lambda value, and the number of actions are exposed as global parameters to be configured. 



### Sample Output

```
Running Q* with lamda: 1.0, max_states_dropout: 10, top_k_actions: 3, semantic_similarity_threshold: 0.6

Selected State: {'value': 'What do I need to do to retire as a millionaire?', 'f_value': 0, 'actions': []}
Expanding Actions: [{'action': 'Identify and prioritize high-growth investments, such as stocks or real estate, to maximize returns over the long-term.', 'q_value': 2.1035156}, {'action': 'Develop a diversified income stream, including sources such as dividend-paying stocks, bonds, and a side hustle, to reduce reliance on a single income source.', 'q_value': 1.828125}, {'action': "Calculate the required savings rate based on the individual's current age, desired retirement age, and expected expenses in retirement.", 'q_value': 0.1439105250238807}]

Selected State: {'f_value': 2.1035156, 'actions': ['Identify and prioritize high-growth investments, such as stocks or real estate, to maximize returns over the long-term.']}
Expanding Actions: [{'action': 'Build multiple income streams, such as starting a side business, investing in dividend-paying stocks, or generating passive income through online platforms, to reduce reliance on a single source of income and accelerate wealth accumulation.', 'q_value': 1.5546875}, {'action': 'Cultivate a mindset of frugality and long-term thinking, avoiding lifestyle inflation and prioritizing needs over wants, to ensure that your wealth-building efforts are sustainable and aligned with your values.', 'q_value': 1.4511719}, {'action': 'Develop a comprehensive financial plan, including a detailed budget, savings strategy, and investment roadmap, tailored to your individual circumstances and goals.', 'q_value': 0.14031994406398182}]

Selected State: {'f_value': 1.828125, 'actions': ['Develop a diversified income stream, including sources such as dividend-paying stocks, bonds, and a side hustle, to reduce reliance on a single income source.']}
Expanding Actions: [{'action': 'Create a comprehensive financial plan, outlining specific goals, timelines, and strategies for achieving millionaire status, including a detailed budget, investment plan, and risk management approach.', 'q_value': 1.8701172}, {'action': 'Invest in yourself by acquiring new skills, knowledge, and certifications that can increase earning potential, such as learning a new language, programming skills, or obtaining a professional certification.', 'q_value': 1.5400391}, {'action': 'Leverage compound interest by starting to save and invest aggressively from an early age, taking advantage of tax-advantaged accounts such as 401(k), IRA, or Roth IRA, and consistently contributing to these accounts over time.', 'q_value': 0.198919273085292}]

Selected State: {'f_value': 3.6982422, 'actions': ['Develop a diversified income stream, including sources such as dividend-paying stocks, bonds, and a side hustle, to reduce reliance on a single income source.', 'Create a comprehensive financial plan, outlining specific goals, timelines, and strategies for achieving millionaire status, including a detailed budget, investment plan, and risk management approach.']}
Expanding Actions: [{'action': 'Invest in a mix of low-cost index funds and ETFs, covering various asset classes, sectors, and geographic regions, to create a diversified investment portfolio that can weather market fluctuations.', 'q_value': 1.9794922}, {'action': 'Maximize tax-advantaged savings vehicles, such as 401(k), IRA, or Roth IRA, to optimize retirement savings and minimize taxes.', 'q_value': 0.2002213062796108}, {'action': 'Develop a habit of consistent saving and investing, setting aside a fixed percentage of income each month, to build wealth over time and take advantage of compound interest.', 'q_value': 0.1834774950463967}]

Selected State: {'f_value': 5.6777344, 'actions': ['Develop a diversified income stream, including sources such as dividend-paying stocks, bonds, and a side hustle, to reduce reliance on a single income source.', 'Create a comprehensive financial plan, outlining specific goals, timelines, and strategies for achieving millionaire status, including a detailed budget, investment plan, and risk management approach.', 'Invest in a mix of low-cost index funds and ETFs, covering various asset classes, sectors, and geographic regions, to create a diversified investment portfolio that can weather market fluctuations.']}
Expanding Actions: [{'action': 'Maximize tax-advantaged accounts, such as 401(k), IRA, or Roth IRA, to optimize savings and reduce tax liabilities, allowing for more efficient wealth accumulation.', 'q_value': 1.8037109}, {'action': 'Cultivate multiple income-generating skills or certifications, such as real estate investing, coding, or digital marketing, to increase earning potential and create additional revenue streams that can accelerate wealth creation.', 'q_value': 1.6884766}, {'action': 'Develop a long-term investment strategy focused on dollar-cost averaging, where a fixed amount of money is invested at regular intervals, regardless of market conditions, to reduce timing risks and emotional decision-making.', 'q_value': 1.5507812}]

Selected State: {'f_value': 7.481445300000001, 'actions': ['Develop a diversified income stream, including sources such as dividend-paying stocks, bonds, and a side hustle, to reduce reliance on a single income source.', 'Create a comprehensive financial plan, outlining specific goals, timelines, and strategies for achieving millionaire status, including a detailed budget, investment plan, and risk management approach.', 'Invest in a mix of low-cost index funds and ETFs, covering various asset classes, sectors, and geographic regions, to create a diversified investment portfolio that can weather market fluctuations.', 'Maximize tax-advantaged accounts, such as 401(k), IRA, or Roth IRA, to optimize savings and reduce tax liabilities, allowing for more efficient wealth accumulation.']}
Expanding Actions: [{'action': 'Build an emergency fund to cover at least 6-12 months of living expenses, which will provide a cushion during market downturns or unexpected events, allowing you to stay the course with your investment strategy and avoid dipping into your investments.', 'q_value': 1.6445312}, {'action': "Develop a long-term investment strategy that focuses on dollar-cost averaging, where you invest a fixed amount of money at regular intervals, regardless of the market's performance, to reduce timing risks and avoid emotional decision-making.", 'q_value': 1.5986328}, {'action': 'Educate yourself on personal finance and investing by reading books, articles, and attending seminars or workshops to gain a deep understanding of the principles and strategies required to achieve millionaire status.', 'q_value': 0.1778871098442052}]

Selected State: {'f_value': 9.1259765, 'actions': ['Develop a diversified income stream, including sources such as dividend-paying stocks, bonds, and a side hustle, to reduce reliance on a single income source.', 'Create a comprehensive financial plan, outlining specific goals, timelines, and strategies for achieving millionaire status, including a detailed budget, investment plan, and risk management approach.', 'Invest in a mix of low-cost index funds and ETFs, covering various asset classes, sectors, and geographic regions, to create a diversified investment portfolio that can weather market fluctuations.', 'Maximize tax-advantaged accounts, such as 401(k), IRA, or Roth IRA, to optimize savings and reduce tax liabilities, allowing for more efficient wealth accumulation.', 'Build an emergency fund to cover at least 6-12 months of living expenses, which will provide a cushion during market downturns or unexpected events, allowing you to stay the course with your investment strategy and avoid dipping into your investments.']}
Expanding Actions: [{'action': "Leverage the power of compound interest by starting to save and invest as early as possible, even if it's just a small amount each month, to take advantage of the exponential growth potential over time.", 'q_value': 1.6513672}, {'action': 'Develop a long-term mindset and avoid getting caught up in get-rich-quick schemes or emotional decision-making, instead focusing on steady, consistent progress towards your financial goals.', 'q_value': 1.4121094}, {'action': 'Educate yourself on personal finance and investing by reading books, articles, and attending seminars or workshops to gain a deeper understanding of the concepts and strategies required to achieve millionaire status.', 'q_value': 0.16233936464932405}]

Selected State: {'f_value': 10.7773437, 'actions': ['Develop a diversified income stream, including sources such as dividend-paying stocks, bonds, and a side hustle, to reduce reliance on a single income source.', 'Create a comprehensive financial plan, outlining specific goals, timelines, and strategies for achieving millionaire status, including a detailed budget, investment plan, and risk management approach.', 'Invest in a mix of low-cost index funds and ETFs, covering various asset classes, sectors, and geographic regions, to create a diversified investment portfolio that can weather market fluctuations.', 'Maximize tax-advantaged accounts, such as 401(k), IRA, or Roth IRA, to optimize savings and reduce tax liabilities, allowing for more efficient wealth accumulation.', 'Build an emergency fund to cover at least 6-12 months of living expenses, which will provide a cushion during market downturns or unexpected events, allowing you to stay the course with your investment strategy and avoid dipping into your investments.', "Leverage the power of compound interest by starting to save and invest as early as possible, even if it's just a small amount each month, to take advantage of the exponential growth potential over time."]}
Expanding Actions: [{'action': 'Build a network of like-minded individuals who share your financial goals and values, such as joining online forums or attending local meetups, to stay motivated, learn from others, and gain access to valuable resources and insights.', 'q_value': 1.6435547}, {'action': "Develop a long-term investment strategy that focuses on dollar-cost averaging, where you invest a fixed amount of money at regular intervals, regardless of the market's performance, to reduce timing risks and avoid emotional decision-making.", 'q_value': 1.5986328}, {'action': 'Educate yourself on personal finance and investing by reading books, articles, and attending seminars or workshops to gain a deeper understanding of the concepts and strategies required to achieve millionaire status.', 'q_value': 0.16233936464932405}]

Selected State: {'f_value': 12.420898399999999, 'actions': ['Develop a diversified income stream, including sources such as dividend-paying stocks, bonds, and a side hustle, to reduce reliance on a single income source.', 'Create a comprehensive financial plan, outlining specific goals, timelines, and strategies for achieving millionaire status, including a detailed budget, investment plan, and risk management approach.', 'Invest in a mix of low-cost index funds and ETFs, covering various asset classes, sectors, and geographic regions, to create a diversified investment portfolio that can weather market fluctuations.', 'Maximize tax-advantaged accounts, such as 401(k), IRA, or Roth IRA, to optimize savings and reduce tax liabilities, allowing for more efficient wealth accumulation.', 'Build an emergency fund to cover at least 6-12 months of living expenses, which will provide a cushion during market downturns or unexpected events, allowing you to stay the course with your investment strategy and avoid dipping into your investments.', "Leverage the power of compound interest by starting to save and invest as early as possible, even if it's just a small amount each month, to take advantage of the exponential growth potential over time.", 'Build a network of like-minded individuals who share your financial goals and values, such as joining online forums or attending local meetups, to stay motivated, learn from others, and gain access to valuable resources and insights.']}
Expanding Actions: [{'action': 'Leverage technology and automation to streamline your finances, such as setting up automatic transfers from your paycheck to your investment accounts, using budgeting apps like Mint or Personal Capital to track your expenses, and taking advantage of robo-advisors or low-cost investment platforms to minimize fees and maximize returns.', 'q_value': 1.9892578}, {'action': 'Develop a long-term mindset and avoid getting caught up in get-rich-quick schemes or emotional decision-making, instead focusing on steady, consistent progress towards your financial goals, and being willing to make sacrifices and adjustments as needed.', 'q_value': 1.5117188}, {'action': 'Educate yourself on personal finance and investing by reading books, articles, and online resources, such as The Simple Path to Wealth, A Random Walk Down Wall Street, and Investopedia, to gain a deep understanding of the principles and strategies required to achieve millionaire status.', 'q_value': 1.1308594}]

Selected State: {'f_value': 14.4101562, 'actions': ['Develop a diversified income stream, including sources such as dividend-paying stocks, bonds, and a side hustle, to reduce reliance on a single income source.', 'Create a comprehensive financial plan, outlining specific goals, timelines, and strategies for achieving millionaire status, including a detailed budget, investment plan, and risk management approach.', 'Invest in a mix of low-cost index funds and ETFs, covering various asset classes, sectors, and geographic regions, to create a diversified investment portfolio that can weather market fluctuations.', 'Maximize tax-advantaged accounts, such as 401(k), IRA, or Roth IRA, to optimize savings and reduce tax liabilities, allowing for more efficient wealth accumulation.', 'Build an emergency fund to cover at least 6-12 months of living expenses, which will provide a cushion during market downturns or unexpected events, allowing you to stay the course with your investment strategy and avoid dipping into your investments.', "Leverage the power of compound interest by starting to save and invest as early as possible, even if it's just a small amount each month, to take advantage of the exponential growth potential over time.", 'Build a network of like-minded individuals who share your financial goals and values, such as joining online forums or attending local meetups, to stay motivated, learn from others, and gain access to valuable resources and insights.', 'Leverage technology and automation to streamline your finances, such as setting up automatic transfers from your paycheck to your investment accounts, using budgeting apps like Mint or Personal Capital to track your expenses, and taking advantage of robo-advisors or low-cost investment platforms to minimize fees and maximize returns.']}
Expanding Actions: [{'action': 'Prioritize living below your means and adopting a frugal lifestyle, avoiding unnecessary expenses and debt, to free up more resources for saving and investing, and to build a strong foundation for long-term wealth accumulation.', 'q_value': 1.7685547}, {'action': 'Develop a long-term perspective and avoid getting caught up in short-term market volatility by focusing on time-tested investment strategies, such as dollar-cost averaging and value investing, to ride out market fluctuations and stay committed to your investment plan.', 'q_value': 1.6445312}, {'action': 'Educate yourself on personal finance and investing by reading books, articles, and online resources, such as The Simple Path to Wealth, A Random Walk Down Wall Street, and Investopedia, to gain a deep understanding of the principles and strategies required to achieve millionaire status.', 'q_value': 1.1308594}]

State with the highest utility has the following actions:
1. Develop a diversified income stream, including sources such as dividend-paying stocks, bonds, and a side hustle, to reduce reliance on a single income source.
2. Create a comprehensive financial plan, outlining specific goals, timelines, and strategies for achieving millionaire status, including a detailed budget, investment plan, and risk management approach.
3. Invest in a mix of low-cost index funds and ETFs, covering various asset classes, sectors, and geographic regions, to create a diversified investment portfolio that can weather market fluctuations.
4. Maximize tax-advantaged accounts, such as 401(k), IRA, or Roth IRA, to optimize savings and reduce tax liabilities, allowing for more efficient wealth accumulation.
5. Build an emergency fund to cover at least 6-12 months of living expenses, which will provide a cushion during market downturns or unexpected events, allowing you to stay the course with your investment strategy and avoid dipping into your investments.
6. Leverage the power of compound interest by starting to save and invest as early as possible, even if it's just a small amount each month, to take advantage of the exponential growth potential over time.
7. Build a network of like-minded individuals who share your financial goals and values, such as joining online forums or attending local meetups, to stay motivated, learn from others, and gain access to valuable resources and insights.
8. Leverage technology and automation to streamline your finances, such as setting up automatic transfers from your paycheck to your investment accounts, using budgeting apps like Mint or Personal Capital to track your expenses, and taking advantage of robo-advisors or low-cost investment platforms to minimize fees and maximize returns.
```