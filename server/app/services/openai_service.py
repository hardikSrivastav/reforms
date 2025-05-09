import os
import openai
import json
from typing import List, Dict, Any

class OpenAIService:
    def __init__(self):
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_metrics(self, goal_description: str) -> List[Dict[str, Any]]:
        """
        Generate survey metrics based on a goal description using OpenAI
        
        Args:
            goal_description: A string describing the survey goal
            
        Returns:
            A list of metric dictionaries with name, type, and description
        """
        prompt = f"""
        You are an expert in survey design. Based on the following survey goal, 
        generate 5-7 appropriate metrics that would help measure this goal effectively.
        
        Survey Goal: "{goal_description}"
        
        For each metric, provide:
        1. A clear, concise name
        2. A type (use one of: likert, text, multiple_choice, rating, boolean)
        3. A brief description explaining what the metric measures
        
        Return the results as a JSON array of objects with the fields: name, type, description.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a survey design expert assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Ensure we got a metrics array
            metrics = data.get("metrics", [])
            if not metrics and isinstance(data, list):
                # Handle case where response might be a direct array
                metrics = data
            
            return metrics
            
        except Exception as e:
            print(f"Error generating metrics: {str(e)}")
            # Return some default metrics in case of error
            return [
                {"name": "Overall Satisfaction", "type": "likert", "description": "General satisfaction with the subject"},
                {"name": "Recommendation Likelihood", "type": "rating", "description": "How likely would you recommend this to others"},
                {"name": "Improvement Areas", "type": "text", "description": "Areas that need improvement"}
            ]

    def generate_question(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a survey question based on a metric using OpenAI
        
        Args:
            prompt: A string containing the metric details and instructions
            
        Returns:
            A dictionary containing the generated question details
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a survey design expert assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content
            data = json.loads(content)
            
            return data
            
        except Exception as e:
            print(f"Error generating question: {str(e)}")
            # Return a default question in case of error
            return {
                "question": "Please rate your overall satisfaction",
                "type": "radio",
                "options": ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"],
                "required": True
            }

# Singleton instance
openai_service = OpenAIService()
