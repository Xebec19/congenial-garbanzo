import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_google_genai import GoogleGenerativeAI

load_dotenv() # Load environment variables from .env file

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API")

search_tool = SerperDevTool()

llm = GoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=GOOGLE_AI_API_KEY)

print("Welcome to AI Researcher and Writer!")
topic = input("Enter the topic: ") # Prompt user to enter a topic

researcher = Agent(
    role="Researcher",
    goal=f"Uncover interesting findings about {topic}",
    verbose=True,
    memory=True,
    backstory=(
    """
    As a researcher, you are committed to uncovering the
    latest and most interesting findings in your field. You
    have a knack for finding hidden gems of information and
    presenting them in an engaging way. Your goal is to
    illuminate the topic at hand, providing insights that are
    both informative and thought-provoking.
    """
    ),
    tools=[search_tool],
    llm=llm,
    allow_delegation=True
)

writer = Agent(
    role="Writer",
    goal=f"Write intuitive article about {topic}",
    verbose=True,
    memory=True,
    backstory=(
    """ 
    As a writer, you are dedicated to crafting engaging
    and informative articles. You have a talent for
    transforming complex ideas into accessible language,
    making them relatable to a wide audience.
    """
    ),
    tools=[search_tool],
    llm=llm,
    allow_delegation=False 
)

research_task = Task(
    description=(
        f"Drive key insights about {topic}."
        "What are the latest trends, technologies, and innovations?"
        "Have a balanced view, considering both the positive and negative aspects."
        "Your report should be well-structured and easy to follow."
    ),
    expected_output="A comprehensive 3 paragraphs long report on the topic",
    tools=[search_tool],
    agent=researcher
)

write_task = Task(
    description=(
        f"Compose an detailed and easy to understand article on {topic}"
        "The article should be engaging and informative, suitable for a general audience."
        "It should be well-structured, with a clear introduction, body, and conclusion."
        "Use markdown formatting for headings, lists, and code snippets where appropriate."
        "The article should be at least 4 paragraphs long and cover the key points from the research report."
    ),
    expected_output=f"A 4 paragraph article on {topic} fomatted as markdown",
    tools=[search_tool],
    agent=writer,
    async_execution=False,
    output_file="blog-post.md"
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential
)

result = crew.kickoff()
print(result)