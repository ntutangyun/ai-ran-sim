import asyncio
import csv
import datetime
import json
import os
import websockets
from schemas import Message, Data, Output, Request, TestOutput
from agents import Agent, Runner


ws_connection = None
ws_uri = "ws://localhost:8765"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
report_file_name = f"test_report_{timestamp}.csv"
report_file_path = os.path.join(os.getcwd(), report_file_name)


instructions = """
    You are an expert in testing a chatbot that is capable of suggesting user with an appropriate model based on their requirement
    You are provided with a scenario and a lot more details about it to test the chatbot.
    Depending on the chatbots questions and responses, drive the chat by providing more details about the scenario.
    If the chatbot has suggested an appropriate model, you can end the conversation by marking the field, 'isDone' to true and setting the suggested model / models in the field `suggested_model`
    The scenario for you to test is: {scenario}
    Incase if the chatbot asks more details about the scenario use this detailed scenario to clarify the queries {detailed_scenario}
    Remember that you are acting as the user for testing and the chatbot which you are testing is the actual assistant
"""


async def getWSConnection():
    global ws_connection
    if ws_connection is None:
        try:
            ws_connection = await websockets.connect(ws_uri, ping_interval=300, ping_timeout=300)
            print("WebSocket connection established.")
        except Exception as e:
            print(f"Error connecting to WebSocket server: {e}")
            return None
    return ws_connection


def get_agent(instruction):
    return Agent(name="Test Agent",
                 instructions=instruction,
                 output_type=TestOutput)

def get_messages(message = None):
    assistant = "assistant"
    user = "user"
    assistant, user = user, assistant
    messages =  [
            Message(
                role=assistant,
                content=(
                    "Monotone\nThank you for choosing our network.\n  \n"
                    "We offer a range of AI services ready-to-deploy across our base stations for your connected user equipments.\n"
                    "Simply describe what applications or use cases you have in mind or you're currently building,\n"
                    "and our AI assistant will help you deploy the services you need across our edge/cloud clusters."
                )
            ),
            Message(
                role=assistant,
                content="Monotone\nChoose an option below to get started."
            ),
            Message(
                role=user,
                content="I wish to request an AI service for my use case."
            ),
            Message(
                role=assistant,
                content="Please describe the AI service you need or the use case you are working on."
            )
        ]

    if message:
        messages.append(message)
    return messages

def create_request_object(messages: list[Message] | None = None, message: Message | None = None):
    if not messages:
        messages = get_messages(message)
    elif message:
        messages.append(message)
    user = "user"
    assistant = "assistant"

    for message in messages:
        message.role =user if message.role == assistant else assistant

    data = Data(current_step="step_service_need_profiling", messages=messages)
    request = Request(layer="intelligence_layer", command="ai_service_pipeline", data=data)
    return request


async def test_ai(agent, messages):
    return await Runner.run(agent, [message.to_dict() for message in messages])

def write_result_to_csv(result:Output):
    file_exists = os.path.isfile(report_file_path)

    with open(report_file_path, 'w' if not file_exists else 'a', newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        field_names = Output.get_field_names()  # Get column names from the class
        if not file_exists:
            csv_writer.writerow(field_names)
        row_values = [getattr(result, field_name) for field_name in field_names]
        csv_writer.writerow(row_values)
        
        
    
async def validate_result(output, scenario):
    prompt = f"""
    You are an expert in model validation. You will be provided with a ground truth (a model or list of models) and a model suggestion from an AI assistant.

    Your tasks are:

    Validation:
     - Determine whether the AI assistant’s suggested model is valid by checking if it matches semantically with any model in the ground truth list.

    Note: Do not perform spelling or formatting checks; but validate the model suggestion with ground truth. 
    For Example: 1. if some other version is suggested that can be considered ok
                 2. 'trpakov/vit-face-expression' - trpakov-vit-face-expression  , these are just the same, so this recommendation is valid.
                 3. 'microsoft/resnet-50' and 'Resnet' should also be considered ok, as the suggestion maps to the same model.
    If at least one model in the AI assistant’s suggestion matches any model in the ground truth, consider the suggestion valid.
    Scoring:
     - Assign a score of 1 if the AI assistant’s suggestion is valid.
     - Assign a score of 0 if it is not.

    Here is the ground truth: {scenario["models"]}
    """

    validate_agent = Agent(name = "validate_agent", instructions=prompt, output_type=int)

    response = await Runner.run(validate_agent, str(output.suggested_model))
    output = Output(scenario = scenario['requirement'], 
                            detailed_scenario = scenario["detailed_requirement"], 
                            ground_truth = str(scenario["models"]), 
                            suggetion = str(output.suggested_model), 
                            score = int(response.final_output))
    write_result_to_csv(output)

async def test_scenario(scenario):
    print("Scenario execution started")
    websocket = await getWSConnection()
    prompt = instructions.replace('{scenario}', scenario['requirement']).replace('{detailed_scenario}',
                                                                               scenario['detailed_requirement'])
    agent = get_agent(prompt)
    messages = get_messages()

    for i in range(10):
        response = await test_ai(agent, messages)
        if response.final_output.isDone:
            await validate_result(response.final_output, scenario=scenario)
            break
        messages.append(Message("assistant", response.final_output.answer))
        request_obj = create_request_object(messages)
        request_json = json.dumps(request_obj.to_dict(), ensure_ascii=False)
        
        await websocket.send(request_json)

        try:
            async for resp in websocket:
                try:
                    data = json.loads(resp)
                except json.JSONDecodeError:
                    print("Received non json message")
                    continue
                if(data['response'] and data['response']['event_type'] == 'message_output_item'):
                    messages.append(Message("user", data['response']['message_output']))
                    break
        except websockets.ConnectionClosed:
            print("Connection closed")
            global ws_connection
            ws_connection = None
            break

        


async def test_scenarios():
    with open("/home/udhay/Documents/ai-ran-sim/backend/test/test_scenarios.json", 'r') as f:
        json_data = json.load(f)
        if (isinstance(json_data, list)):
            for scenario in json_data:
                await test_scenario(scenario)





async def connect_to_server():
    uri = "ws://localhost:8765"  # Replace with your WebSocket server URL
    request_obj = create_request_object()
    request_json = json.dumps(request_obj.to_dict(), ensure_ascii=False)

    async with websockets.connect(uri) as websocket:
        # Send the JSON request
        await websocket.send(request_json)
        print("Request sent to the server.")

        try:
            # Continuously listen for streamed responses
            async for message in websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    print("Received non-JSON message:", message)
                    continue
                if (data['response']['event_type'] == 'message_output_item'):
                    print(data['response']['message_output'])
        except websockets.ConnectionClosed:
            print("Connection closed by server.")


# Run the async function
asyncio.run(test_scenarios())