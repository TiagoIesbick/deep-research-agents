import asyncio
from questioner_agent import InteractiveQuestioner


async def example_questioning_session():
    """Example of how to use the InteractiveQuestioner."""
    
    # Create the interactive questioner
    questioner = InteractiveQuestioner(max_questions=3)
    
    # Start with an initial query
    initial_query = "I want to learn about machine learning"
    print(f"User Query: {initial_query}")
    print("-" * 50)
    
    # Ask the first question
    first_question = await questioner.start_questioning(initial_query)
    print(f"Question 1: {first_question.question}")
    print(f"Reasoning: {first_question.reasoning}")
    print()
    
    # Simulate user answering
    user_answer_1 = "I'm a beginner with no programming experience"
    print(f"User Answer: {user_answer_1}")
    print("-" * 50)
    
    # Ask follow-up question
    second_question = await questioner.ask_follow_up(user_answer_1)
    if second_question:
        print(f"Question 2: {second_question.question}")
        print(f"Reasoning: {second_question.reasoning}")
        print()
        
        # Simulate user answering
        user_answer_2 = "I want to understand the basics and maybe build simple models"
        print(f"User Answer: {user_answer_2}")
        print("-" * 50)
        
        # Ask final follow-up question
        third_question = await questioner.ask_follow_up(user_answer_2)
        if third_question:
            print(f"Question 3: {third_question.question}")
            print(f"Reasoning: {third_question.reasoning}")
            print()
            
            # Simulate user answering
            user_answer_3 = "I have about 2-3 hours per week to study"
            print(f"User Answer: {user_answer_3}")
            print("-" * 50)
            
            # Get final summary
            final_summary = await questioner.get_final_summary()
            print("FINAL UNDERSTANDING:")
            print(f"Original Query: {final_summary.original_query}")
            print(f"Refined Understanding: {final_summary.refined_understanding}")
            print("Key Clarifications:")
            for clarification in final_summary.key_clarifications:
                print(f"  - {clarification}")
    
    # Show progress
    progress = questioner.get_questioning_progress()
    print(f"\nProgress: {progress['questions_asked']}/{progress['max_questions']} questions asked")


async def interactive_session():
    """Interactive session where user can actually answer questions."""
    
    questioner = InteractiveQuestioner(max_questions=3)
    
    # Get initial query from user
    initial_query = input("What would you like to learn about? ")
    print(f"\nStarting questioning session for: {initial_query}")
    print("-" * 50)
    
    # Ask the first question
    first_question = await questioner.start_questioning(initial_query)
    print(f"Question 1: {first_question.question}")
    print(f"Reasoning: {first_question.reasoning}")
    print()
    
    # Get user's answer
    user_answer_1 = input("Your answer: ")
    print("-" * 50)
    
    # Ask follow-up question
    second_question = await questioner.ask_follow_up(user_answer_1)
    if second_question:
        print(f"Question 2: {second_question.question}")
        print(f"Reasoning: {second_question.reasoning}")
        print()
        
        # Get user's answer
        user_answer_2 = input("Your answer: ")
        print("-" * 50)
        
        # Ask final follow-up question
        third_question = await questioner.ask_follow_up(user_answer_2)
        if third_question:
            print(f"Question 3: {third_question.question}")
            print(f"Reasoning: {third_question.reasoning}")
            print()
            
            # Get user's answer
            user_answer_3 = input("Your answer: ")
            print("-" * 50)
            
            # Get final summary
            final_summary = await questioner.get_final_summary()
            print("FINAL UNDERSTANDING:")
            print(f"Original Query: {final_summary.original_query}")
            print(f"Refined Understanding: {final_summary.refined_understanding}")
            print("Key Clarifications:")
            for clarification in final_summary.key_clarifications:
                print(f"  - {clarification}")


if __name__ == "__main__":
    print("Interactive Questioning Agent Demo")
    print("=" * 50)
    print()
    
    choice = input("Choose demo mode:\n1. Example session (predefined answers)\n2. Interactive session (you answer)\nEnter 1 or 2: ")
    
    if choice == "1":
        asyncio.run(example_questioning_session())
    elif choice == "2":
        asyncio.run(interactive_session())
    else:
        print("Invalid choice. Running example session...")
        asyncio.run(example_questioning_session())
