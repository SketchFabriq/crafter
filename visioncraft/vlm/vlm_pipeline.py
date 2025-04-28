from visioncraft.vlm.molmo_api import call_molmo, draw_point_on_image, draw_multiple_points_on_image
from visioncraft.vlm.vlm_selector import *
import json
import re


class VLMPipeline:
    def __init__(self, model_name, openai_api_key=None):
        self.gptmodel = ExternalModel(model_name, openai_api_key)
        self.current_image = None
        self.goal_image = None
        self.prompt = None
        self.result = None

    def set_current_image(self, image_path):
        self.current_image = image_path

    def set_goal_image(self, image_path):
        self.goal_image = image_path

    def set_prompt(self, prompt):
        self.prompt = prompt

    def run_pipeline(self):
        # Could count number of blocks and have it dynamic, here is hardcoded to 4
        prompt = """
        You are an agent assisting me in taking given images, one of a goal build and the other of the current environment. 
        Given the current environment, you will help me direct my robotic arm to build the goal construction. 
        We will deconstruct this setup step by step.
        First, on the first given image, you need to deconstruct this into a buildible structure. 
        The available coordinates for constructions are 10x10, therefore the coordinates of the blocks should be between 0 and 10 for x and y. 0,0 is bottom left. The blocks do not necessarily have to touch each other,
        but the overall structure should be buildible and recognizable. Take into account the places where no blocks should be so that the structure is recognizable.
        
        Now, given the second image of the current environment , look for the blocks at the left of the image. 
        You will need to take them and place them as you previously planned in the construction zone.

        Given 4 blocks, I want you to give me step by step where to pick the blocks and where to place them. You do not have to use all the blocks if the structure doesn't require it.
        You may only answer with the format ACTION X : { pos_init : {x:val, y:val}, pos_finale : {x:val, y:val}, notice : str } 

        Action X being the step number, and notice being a string that will help me understand what to do (Ex. approach from the top).

        You may chose the initial blocks positions as a grid, 0,0 being the bottom left block. You can assume that these given pos_init for the blocks will not impact the coordinate system for the 
        built object, its only used to help me understand which blocks to pick up.
        """
        if self.current_image is None or self.goal_image is None:
            raise ValueError("Current image and goal image must be set before starting the pipeline.")
        
        # 1. Send the base prompt to openai
        print("Sending prompt to OpenAI...")
        base_response = self.gptmodel.send_images(prompt, [self.goal_image, self.current_image])
        print("Base response received.")
        print(base_response)

        # 2. Get json response out of base response with another call.
        json_prompt = "Based on this given text, can you give me the actions as a json file only ? No other words, only json. Here is an example of the format expected : {actions : [{pos_init : [], ...}, ...] } Here is the text : " + base_response
        json_response = self.gptmodel.query(json_prompt)
        print("JSON response received.")
        print(json_response)
        # verify if json response is valid
       

        # Get all points with molmo
        molmo_prompt = "Point to all the blocks in the left part of the image"
        molmo_result = call_molmo(self.current_image, molmo_prompt)
        print("received result : ", molmo_result)
        matches = re.findall(r'x\d+="([\d.]+)" y\d+="([\d.]+)"', molmo_result)
    
        # Convert to list of tuples of floats
        coordinates = [(float(x), float(y)) for x, y in matches]
        print(coordinates)
        draw_multiple_points_on_image(self.current_image, "out_multiple.png",coordinates) # All points on one image

        # Here we save a picture for all coordinates
        for i in range(len(coordinates)):
            draw_point_on_image(self.current_image,"out_"+str(i)+".png", coordinates[i][0],  coordinates[i][1])
        
        # TODO: Here we can verify if the points are ok, since we saved each individual point as a different image, or one image with all points
        # If we have time, we can add the loop verification system, sending back the image to openai and asking if the points are correct

        # We have the actions and point coordinates. now we should link them up
        second_prompt = "Given this json : " + str(json_response) + " and this array : " +str(coordinates) + ", can you help me associate the pos_init positions with a given coordinate ? They should not necessarily be in the same order, but should make sense as to where they are placed based on their values. Your answer should be in a json format like the one given, but with the pos_init values replaced. Add no comment, just raw json."
        second_response = self.gptmodel.query(second_prompt)
        print(second_response)
        print("Verifying JSON response...")
        try:
            json_match = re.search(r"\{.*\}", second_response, re.DOTALL)
            if json_match:
                json_response = json_match.group(0)  # Extract the JSON part
                json_response = json.loads(json_response)
            print("JSON response is valid.")
        except json.JSONDecodeError:
            print("JSON response is invalid.")
            return

        return json_response

if __name__ == "__main__":
    pipeline = VLMPipeline(model_name="gpt-4o", openai_api_key=os.environ.get("OPENAI_API_KEY"))
    pipeline.set_current_image("resources/test1.jpg")
    pipeline.set_goal_image("resources/simple_house.png")
    pipeline.run_pipeline()