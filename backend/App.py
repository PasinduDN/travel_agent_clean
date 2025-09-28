from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import load_tools
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from operator import add as add_messages
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import re

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("OPENWEATHERMAP_API_KEY")

chat_sessions = {}

app = FastAPI()  # Your existing FastAPI app

# Add this immediately after creating `app`
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MOCK_POSTS = [
    {
        "id": 1,
        "imageUrl": "https://picsum.photos/300/200?random=1",
        "caption": "Exploring the tea plantations in Nuwara Eliya 🌱☕",
        "hashtags": ["#NuwaraEliya", "#SriLanka", "#Tea"],
        "likes": 120,
        "username": "traveler_anna"
    },
    {
        "id": 2,
        "imageUrl": "https://picsum.photos/300/200?random=2",
        "caption": "Beach vibes at Unawatuna 🏖️",
        "hashtags": ["#Beach", "#SriLanka", "#Unawatuna"],
        "likes": 245,
        "username": "beachlover"
    },
    # ...add more posts (at least 10–15 covering destinations, food, activities, etc.)
]

DISTRICTS = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya",
    "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar",
    "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
    "Monaragala", "Ratnapura", "Kegalle"
]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

#Define AgentState
class AgentState(TypedDict):
    # This annotation tells LangGraph to append new messages to the existing list
    messages: list
    preferences_text: str
    locations: list

# Define the request body
class PreferencesRequest(BaseModel):
    preferences_text: str
    locations: List[str]
    start_date: str  # Use YYYY-MM-DD format
    end_date: str
    liked_posts: List[str] = []

class FirstMessageRequest(BaseModel):
    preferences_text: str

@tool
def destination_tool(preferences: str) -> str:
    """Suggest 3-5 destinations in Sri Lanka based on user preferences."""
    system_prompt = """
    You are a travel assistant for Sri Lanka.
    Based on the user's preferences, suggest 3-5 destinations.
    Respond only with a list of destinations separated by commas.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User preferences: {preferences}")
    ]
    response = llm.invoke(messages)
    return response.content


@tool
def accommodation_tool(preferences: str) -> str:
    """
    Generate recommended accommodations in Sri Lanka based on user preferences.
    """
    system_prompt = """
    You are a travel assistant for Sri Lanka.
    Based on the user's accommodation preferences, suggest hotels, resorts, lodges, or guesthouses.
    Include 3-5 options.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User preferences: {preferences}")
    ]
    response = llm.invoke(messages)
    return response.content

@tool
def food_tool(preferences: str) -> str:
    """
    Suggest food options in Sri Lanka based on user preferences.
    """
    system_prompt = """
    You are a Sri Lanka travel assistant.
    Suggest 3-5 restaurants, food types, or dishes that match the user's food preferences.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User preferences: {preferences}")
    ]
    response = llm.invoke(messages)
    return response.content

@tool
def activity_tool(preferences: str) -> str:
    """
    Suggest activities in Sri Lanka based on user preferences.
    """
    system_prompt = """
    You are a Sri Lanka travel assistant.
    Based on the user's activity preferences, suggest 3-5 relevant activities or tours.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User preferences: {preferences}")
    ]
    response = llm.invoke(messages)
    return response.content

@tool
def weather_tool(location: str, travel_date: str, preferences: str = "") -> str:
    """
    Suggest locations in Sri Lanka based on weather preferences.
    """
    weather_api = OpenWeatherMapAPIWrapper()
    forecast = weather_api.get_weather(location, travel_date)
    
    # Use LLM to filter locations based on forecast + user preferences
    system_prompt = f"""
    You are a travel assistant for Sri Lanka.
    The forecast for {location} on {travel_date} is: {forecast}.
    Based on the user's weather preferences ({preferences}), suggest 1-3 locations suitable for visiting.
    """
    messages = [SystemMessage(content=system_prompt)]
    response = llm.invoke(messages)
    return response.content

@tool("update_locations", return_direct=True)
def update_locations(preferences: str, current_locations: list = []) -> list:
    """
    Extract valid Sri Lankan districts from user preferences and update the list.
    """
    new_locations = [d for d in DISTRICTS if d.lower() in preferences.lower()]
    updated = list(set(current_locations + new_locations))
    return updated

# Utility function to clean up text
def clean_text(text: str) -> str:
    # Remove emojis (this regex covers a wide range of emojis)
    text = re.sub(r'[^\w\s#]', '', text)
    # Remove newlines and extra spaces
    text = text.replace("\n", " ").strip()
    return text.lower()

# Utility function to extract relevant keywords from query
def extract_keywords(query: str) -> list:
    # Here, you can define more specific keywords or use some NLP techniques
    keywords = ['seafood', 'water sports', 'nature', 'adventure', 'beach', 'mountain', 'cultural', 'historical', 'eco', 'luxury']
    # Extract words that are in the list of keywords from the query
    return [word for word in query.lower().split() if word in keywords]

@tool("instagram_posts", return_direct=True)
def fetch_instagram_posts(query: str, limit: int = 5) -> list:
    """
    Fetch Instagram posts related to a query using the Graph API.
    Returns a list of posts with id, imageUrl, caption, hashtags, username, permalink.
    """
    # url = f"https://graph.facebook.com/v19.0/17841454539810512/media"
    url = f"https://graph.instagram.com/17841454539810512/media"
    params = {
        "fields": "id,caption,media_url,permalink,media_type,timestamp,username",
        "access_token": "IGAAL77Biu4LJBZAFNHWGtGRFNEelJlZAlE0ajZACUEhVVFpBYUQ0Wm5YcElMLTdKWmdnM0FTYUhTNHJ1b2lxNE81V09jblZATWUFZAUGhZAaXUxM3hER0k2ZATY2U1h6TXZAOMUxFZA1ZA4czg0UlhJazFwaTQ4eTBLRGwwbFBUWFl1T0xpbwZDZD",
        "limit": limit
    }
    try:
        response = requests.get(url, params=params)
        print("Instagram API response:", response.json())  # Debugging the response

        if response.status_code != 200:
            print("Instagram API error:", response.json())  # Check for API error
            return []

        data = response.json().get("data", [])
        print(f"Total posts fetched: {len(data)}")

        # Clean query and extract relevant keywords
        cleaned_query = clean_text(query)  # Clean the query
        print(f"Cleaned query: {cleaned_query}")
        keywords = extract_keywords(query)  # Extract keywords from the query
        print(f"Extracted keywords: {keywords}")

        filtered = []
        for post in data:
            caption = post.get("caption", "")
            cleaned_caption = clean_text(caption)  # Clean the caption
            print(f"Checking cleaned caption: {cleaned_caption}")

            # Check if any keyword from the query is in the cleaned caption
            if any(keyword in cleaned_caption for keyword in keywords):
                print(f"Post matched: {caption}")
                filtered.append(post)

        # Debugging output
        print(f"Filtered posts: {filtered}")
        
        return [
            {
                "id": post["id"],
                "imageUrl": post.get("media_url"),
                "caption": post.get("caption", ""),
                "hashtags": [tag for tag in post.get("caption", "").split() if tag.startswith("#")],
                "likes": 0,
                "username": post.get("username", "unknown"),
                "permalink": post.get("permalink"),
            }
            for post in filtered
        ]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching posts: {e}")
        return []

#LLM Agent Function
def call_llm(state: AgentState) -> AgentState:
    """
    Calls LLM to process the user's preferences text
    and decide which tools to call.
    """
    # The system prompt can be added at the beginning of the chat session
    # For simplicity here, we ensure it's present before calling the LLM
    messages = state['messages']
    
    system_prompt = """
    You are an intelligent AI travel assistant for tourists in Sri Lanka.
    You have access to the following tools:
    - destination_tool
    - accommodation_tool
    - food_tool
    - activity_tool
    - weather_tool
    Based on the full conversation history, decide if a tool is needed or if you can answer the user.
    If you need to call a tool, do so and then synthesize the results into a final answer. Provide all information at once without any intermediate responses like 'Please hold on for a moment.' Ensure the answer is comprehensive and helpful, incorporating results from the tools.
    """
    
    # Check if system prompt is already there to avoid adding it multiple times
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages

    # Call LLM
    message = llm.invoke(messages)
    
    # The 'add_messages' reducer will append this new message to the state
    return {'messages': [message]}


#Tool Execution Function 
def take_action(state: AgentState) -> AgentState:
    tools_dict = {
        "destination_tool": destination_tool,
        "accommodation_tool": accommodation_tool,
        "food_tool": food_tool,
        "activity_tool": activity_tool,
        "weather_tool": weather_tool,
        "update_locations": update_locations,
        "instagram_posts": fetch_instagram_posts
    }
    
    results = []
    tool_calls = getattr(state['messages'][-1], 'tool_calls', [])
    
    for t in tool_calls:
        tool_name = t['name']
        args = t['args']
        if tool_name in tools_dict:
            result = tools_dict[tool_name].invoke(**args)
        else:
            result = f"Tool {tool_name} not found."
        results.append(ToolMessage(tool_call_id=t.get('id', ""), name=tool_name, content=str(result)))
    
    return {'messages': results}

workflow = StateGraph(AgentState)
workflow.add_node("llm", call_llm)
workflow.add_node("tool_agent", take_action)

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0

workflow.add_edge(START, "llm")
workflow.add_conditional_edges("llm", should_continue, {True: "tool_agent", False: END})
workflow.add_edge("tool_agent", "llm")

compiled_workflow = workflow.compile()

@app.post("/api/process_preferences/{session_id}")
async def process_preferences(session_id: str, request: PreferencesRequest):
    if (
        not request.preferences_text.strip()
        or not request.locations
        or not request.start_date.strip()
        or not request.end_date.strip()
    ):
        raise HTTPException(status_code=400, detail="Please provide all fields.")
    
    # Ensure session container exists (messages + locations)
    if session_id not in chat_sessions or not isinstance(chat_sessions[session_id], dict):
        chat_sessions[session_id] = {"messages": [], "locations": []}

    # Merge onboarding locations into session (first call)  
    if not chat_sessions[session_id]["locations"]:
        chat_sessions[session_id]["locations"] = list(dict.fromkeys(request.locations))  # unique+keep order

    prefs_lower = request.preferences_text.lower()
    relevant_posts = [
        post for post in MOCK_POSTS
        if any(tag.lower().replace("#", "") in prefs_lower for tag in post["hashtags"])
    ]

    if not relevant_posts:
        relevant_posts = fetch_instagram_posts.invoke({
            "query": request.preferences_text,
            "limit": 5
        })

    # Build the text the LLM sees
    locations_str = ", ".join(chat_sessions[session_id]["locations"])
    full_preferences = (
        f"Locations: {locations_str}. "
        f"Travel Dates: {request.start_date} to {request.end_date}. "
        f"Preferences: {request.preferences_text}"
    )

    # 1) Always run location-extraction tool to capture any newly mentioned districts
    updated_locations = update_locations.invoke({
        "preferences": request.preferences_text,
        "current_locations": chat_sessions[session_id]["locations"]
    })
    chat_sessions[session_id]["locations"] = updated_locations

    # 2) Add the human message and run the graph
    chat_sessions[session_id]["messages"].append(HumanMessage(content=full_preferences))
    state = {
        "messages": chat_sessions[session_id]["messages"],
        "preferences_text": full_preferences,
        "locations": chat_sessions[session_id]["locations"],
    }
    # print(request.dict())
    result = compiled_workflow.invoke(state)

    # Save the latest model/tool message
    chat_sessions[session_id]["messages"].append(result['messages'][-1])

    print("*************************************")
    print("relevant_posts : ", relevant_posts)
    # 3) Return both the AI text and the updated locations for the map
    return {
        "result": result['messages'][-1].content,
        "locations": chat_sessions[session_id]["locations"],
        "posts": relevant_posts
    }

# Find this endpoint and change the type hint for 'request'

@app.post("/api/process_preferences_forFirstSendMessage/{session_id}")
async def process_preferences_for_first_send_message(session_id: str, request: FirstMessageRequest): # Changed from PreferencesRequest
    """
    Receives user travel preferences, processes them with an LLM,
    and returns a suggested travel plan.
    """
    print(f"Received request for session_id: {session_id}")
    print(f"Preferences: {request.preferences_text}")

    # Basic input validation
    if not request.preferences_text or not request.preferences_text.strip():
        raise HTTPException(status_code=400, detail="preferences_text cannot be empty.")

    # Define the persona and instructions for the LLM
    system_prompt = """
    You are a creative and helpful Sri Lankan travel assistant.
    Based on the user's preferences, generate a concise, engaging, and
    helpful travel suggestion. Mention 2-3 specific places or activities.
    Format your response as a single paragraph.
    """

    # Create the message list to send to the LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=request.preferences_text) # Use the field from the new model
    ]

    try:
        # Get the response from the LLM
        llm_response = llm.invoke(messages)
        ai_content = llm_response.content

        # Return the response to the frontend
        return {"response": ai_content}

    except Exception as e:
        # Handle potential errors from the LLM API call
        print(f"Error calling LLM: {e}")
        raise HTTPException(status_code=500, detail="Failed to get a response from the AI model.")

# # Then at the bottom, for testing:
# test_preferences = "Beach, Eco-Lodge, Local Sri Lankan, Hiking & Nature Trails"

# result = compiled_workflow.invoke({
#     "messages": [HumanMessage(content="Process preferences")],
#     "preferences_text": test_preferences
# })

# print(result)  # see the full output structure
