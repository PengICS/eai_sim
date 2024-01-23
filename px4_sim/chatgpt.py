import openai
import re
import argparse
from src.isaacsim_wrapper import *
import math
import numpy as np
import os
import json
import time
import _thread

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="px4_sim/prompts/airsim_basic.txt")
parser.add_argument("--sysprompt", type=str, default="px4_sim/system_prompts/airsim_basic.txt")
args = parser.parse_args()

with open("px4_sim/config/config.json", "r") as f:
    config = json.load(f)

print("Initializing ChatGPT...")
openai.api_key = config["OPENAI_API_KEY"]
openai.api_base = "https://oneapi.xty.app/v1"

with open(args.sysprompt, "r") as f:
    sysprompt = f.read()

chat_history = [
    {
        "role": "system",
        "content": sysprompt
    },
    {
        "role": "user",
        "content": "move 10 units up"
    },
    {
        "role": "assistant",
        "content": """```python
aw.fly_to([aw.get_drone_position()[0], aw.get_drone_position()[1], aw.get_drone_position()[2]+10])
```

This code uses the `fly_to()` function to move the drone to a new position that is 10 units up from the current position. It does this by getting the current position of the drone using `get_drone_position()` and then creating a new list with the same X and Y coordinates, but with the Z coordinate increased by 10. The drone will then fly to this new position using `fly_to()`."""
    }
]


def ask(prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return chat_history[-1]["content"]


print(f"Done.")

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)


def extract_python_code(content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None


class colors:  # You may need to change color settings
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"


print(f"Initializing isaac Sim...")
# aw = IsaacSimWrapper(50040)
# asyncio.get_event_loop().run_until_complete(aw.connect(14542))

aw1 = IsaacSimWrapper(50041)
asyncio.get_event_loop().run_until_complete(aw1.connect(14543))

aw2 = IsaacSimWrapper(50040)
asyncio.get_event_loop().run_until_complete(aw2.connect(14542))

# aw.arm()
# asyncio.get_event_loop().run_until_complete(asyncio.sleep(2))

aw1.arm()
asyncio.get_event_loop().run_until_complete(asyncio.sleep(2))

aw2.arm()
asyncio.get_event_loop().run_until_complete(asyncio.sleep(2))


# aw.takeoff()
# asyncio.get_event_loop().run_until_complete(asyncio.sleep(2))
# aw1.takeoff()
# aw2.takeoff()
# try:
#    _thread.start_new_thread( aw.takeoff(), ("Thread-1", 2, ) )
#    _thread.start_new_thread( aw1.takeoff(), ("Thread-2", 4, ) )
# except:
#    print ("Error: unable to start thread")




print(f"Done.")

# current_position = aw.get_position()
# new_position = [current_position[0] + 1, current_position[1], current_position[2]]
# aw.fly_to(new_position)

with open(args.prompt, "r") as f:
    prompt = f.read()

ask(prompt)
print("Welcome to the isaac Sim chatbot! I am ready to help you with your isaacSim questions and commands.")

while True:
    question = input(colors.YELLOW + "iassc Sim> " + colors.ENDC)

    if question == "!quit" or question == "!exit":
        break

    if question == "!clear":
        os.system("cls")
        continue

    response = ask(question)

    print(f"\n{response}\n")

    code = extract_python_code(response)
    if code is not None:
        print("Please wait while I run the code in isaac sim...")
        exec(extract_python_code(response))
        print("Done!\n")

