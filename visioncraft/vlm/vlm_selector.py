import os
import ollama
from openai import OpenAI
import base64


class BaseModel:
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.api_key = api_key



class ExternalModel(BaseModel):
    def __init__(self, model_name, api_key=None):
        super().__init__(model_name, api_key)
        self.setup_api()

    def setup_api(self):
        if "gpt" in self.model_name.lower():
            self.client = OpenAI(api_key=self.api_key)
            print(
                f"Using OpenAI API with model {self.model_name}"
            )
        elif "claude" in self.model_name.lower():
            raise NotImplementedError
        else:
            raise ValueError(f"API model {self.model_name} not supported.")

    def query(self, prompt):
        result = None
        if "gpt" in self.model_name.lower():
            response = self.client.responses.create(model=self.model_name, input=prompt)
            result = response.output_text
        elif "claude" in self.model_name.lower():
            raise NotImplementedError
        return result

    def send_images(self, prompt, image_paths):
        if "gpt" in self.model_name.lower():
            encoded_images = []
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                    encoded_images.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded_image}",
                    }
                )
             # Construct the input payload with the prompt and images
            input_payload = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}] + encoded_images,
                }
            ]
            # Send the request to the OpenAI API
            response = self.client.responses.create(
                model=self.model_name,
                input=input_payload,
            )
            return response.output_text
        else:
            raise NotImplementedError("Image sending is not supported for this model.")


class LocalModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        # Note : You need to pull the model before using it using the following command
        # ollama.pull(model_name)
        self.available_models = [
            "deepseek-r1",
            "llama3.2",
            "mistral",
            "llava:7b",
            "gemma3",
            "gemma3:1b",
        ]
        if model_name not in self.available_models:
            raise ValueError(f"Local model {model_name} not supported.")

    def query(self, prompt):
        result = None
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            result = response["message"]["content"]
        except Exception as e:
            print(
                f"Error: {e}. Did you pull the model {self.model_name} before using it?"
            )
        return result

    def send_images(self, prompt, image_paths):
        encoded_images = []
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                encoded_images.append(encoded_image)
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            images=encoded_images,  # Pass base64 encoded image data at top level
            options={
                "temperature": 0.1
            },  # Lower temperature for more consistent output
        )
        return response["response"].strip()


def get_model(model_name, model_type, api_key=None):
    if model_type == "external":
        return ExternalModel(model_name, api_key)
    elif model_type == "local":
        return LocalModel(model_name)
    else:
        raise ValueError(f"Model type {model_type} not supported.")


if __name__ == "__main__":
    prompt = """
    You are an agent assisting me in taking given images, one of a goal buidl and the other of the current environment. 
    Given the current environment, you will help me direct my robotic arm to build the goal construction. 
    We will deconstruct this setup step by step.
    First, on the first given image, you need to deconstruct this into a buildible structure of 10 blocks. 
    The available coordinates for constructions are 10x10, therefore the coordinates of the blocks should be between 0 and 10 for x and y.
    Now, given the second image of the current environment , look for the blocks at the lest of the image. 
    You will need to take them and place them as you previously planned in the construction zone. You can assume the bottom corner of the table being coordinate (0,0).
    
    Given 10 number of blocks, I want you to give me step by step where to pick the blocks and where to place them.
    You may only answer with the format ACTION X : { pos_init : {x, y}, pos_finale : {x, y}, notice : str } 

    Action X being the step number, and notice being a string that will help me understand what to do (Ex. approach from the top).
    """
    prompt = "given the image on the right only, can you give me the coordinates of the most bottom left block in the image? you can use the original image size or assume it is 100x100. You may only answer with the format {x, y}."
    model = get_model("gpt-4o", "external", api_key=os.environ.get("OPENAI_API_KEY"))
    #print(model.query("Hello!"))
    print(model.send_images(prompt, ["resources/house.png", "resources/test1.jpg"]))
    #model = get_model("llava:7b", "local")
    #print(model.send_image(prompt, ["resources/house.png", "resources/test1.jpg"]))
