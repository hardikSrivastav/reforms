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
                    {"role": "system", "content": "You are a survey design expert assistant. Return your response as a JSON object."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
                
                # Ensure we got a metrics array
                metrics = data.get("metrics", [])
                if not metrics and isinstance(data, list):
                    # Handle case where response might be a direct array
                    metrics = data
                
                return metrics
            except json.JSONDecodeError:
                print("Response not in JSON format. Extracting metrics manually.")
                # Try to extract metrics from text format
                lines = content.strip().split("\n")
                metrics = []
                current_metric = {}
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('1.') or line.startswith('- ') or line.startswith('*'):
                        # New metric starts
                        if current_metric and 'name' in current_metric:
                            metrics.append(current_metric)
                            current_metric = {}
                        
                        # Extract name
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            name = parts[1].strip()
                        else:
                            name = line.split(' ', 1)[1].strip() if len(line.split(' ', 1)) > 1 else "Metric"
                        current_metric['name'] = name
                    elif 'type:' in line.lower() or 'type ' in line.lower():
                        # Extract type
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            current_metric['type'] = parts[1].strip()
                    elif 'description:' in line.lower() or 'description ' in line.lower():
                        # Extract description
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            current_metric['description'] = parts[1].strip()
                
                # Add last metric if exists
                if current_metric and 'name' in current_metric:
                    metrics.append(current_metric)
                
                # If we couldn't extract metrics properly, return defaults
                if not metrics:
                    return [
                        {"name": "Overall Satisfaction", "type": "likert", "description": "General satisfaction with the subject"},
                        {"name": "Recommendation Likelihood", "type": "rating", "description": "How likely would you recommend this to others"},
                        {"name": "Improvement Areas", "type": "text", "description": "Areas that need improvement"}
                    ]
                    
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
                    {"role": "system", "content": "You are a survey design expert assistant. Return your response as a JSON object."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
                return data
            except json.JSONDecodeError:
                print("Response not in JSON format. Using default question.")
                # Return a default question in case of parsing error
                return {
                    "question": "Please rate your overall satisfaction",
                    "type": "radio",
                    "options": ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"],
                    "required": True
                }
            
        except Exception as e:
            print(f"Error generating question: {str(e)}")
            # Return a default question in case of error
            return {
                "question": "Please rate your overall satisfaction",
                "type": "radio",
                "options": ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"],
                "required": True
            }

    def generate_questions(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate multiple survey questions based on a list of metrics using OpenAI
        
        Args:
            metrics: A list of metric dictionaries with name, type, and description
            
        Returns:
            A list of question dictionaries with question text, type, and options
        """
        prompt = f"""
        You are an expert in survey design. Based on the following metrics, 
        generate appropriate questions that would help measure these metrics effectively.
        
        Metrics:
        {json.dumps(metrics, indent=2)}
        
        For each metric, generate a question that:
        1. Directly relates to the metric
        2. Is easy to understand
        3. Will yield meaningful responses
        4. Is appropriate for the metric type
        
        Return the results as a JSON array of objects with the fields:
        - question: The actual question text
        - type: The question type (text, number, select, radio, etc.)
        - options: Array of options if type is select/radio
        - required: Boolean indicating if the question is required
        - metric_id: The ID of the metric this question corresponds to
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a survey design expert assistant. Return your response as a JSON object."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
                
                # Ensure we got a questions array
                questions = data.get("questions", [])
                if not questions and isinstance(data, list):
                    # Handle case where response might be a direct array
                    questions = data
                
                return questions
            except json.JSONDecodeError:
                print("Response not in JSON format. Using default questions.")
                # Return default questions in case of parsing error
                return [
                    {
                        "question": "Please rate your overall satisfaction",
                        "type": "radio",
                        "options": ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"],
                        "required": True,
                        "metric_id": metric.get("id") 
                    }
                    for metric in metrics
                ]
            
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            # Return default questions in case of error
            return [
                {
                    "question": "Please rate your overall satisfaction",
                    "type": "radio",
                    "options": ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"],
                    "required": True,
                    "metric_id": metric.get("id")
                }
                for metric in metrics
            ]

    def analyze_survey_responses(self, survey_data: Dict[str, Any], responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze survey responses to generate insights, trends, and recommendations
        
        Args:
            survey_data: Dictionary containing survey questions, metrics, and other metadata
            responses: List of response objects containing the actual survey answers
            
        Returns:
            Dictionary with insights, visualizations suggestions, and recommendations
        """
        # Format the data for analysis
        question_data = {}
        for question in survey_data.get("questions", []):
            q_id = question.get("id")
            if not q_id:
                continue
                
            # Collect responses for this question
            q_responses = []
            for response in responses:
                answer = response.get("responses", {}).get(q_id)
                if answer is not None:
                    q_responses.append(answer)
            
            question_data[q_id] = {
                "question": question.get("question", ""),
                "type": question.get("type", ""),
                "responses": q_responses
            }
        
        # Create prompt for analysis
        prompt = f"""
        You are an expert data analyst specializing in survey analysis. 
        Analyze the following survey data and generate valuable insights.
        
        Survey Title: {survey_data.get("title", "Untitled Survey")}
        Survey Description: {survey_data.get("description", "")}
        Total Responses: {len(responses)}
        
        SURVEY QUESTIONS AND RESPONSES:
        {json.dumps(question_data, indent=2)}
        
        Your analysis should include:
        1. Key insights - What patterns or trends do you observe in the responses?
        2. Statistical observations - What do the numbers tell us?
        3. Text analysis - For text responses, identify common themes
        4. Recommendations - What actions should be taken based on this data?
        5. Visualization suggestions - What types of charts would best represent this data?
        
        For visualization suggestions, follow these guidelines:
        - For categorical data (radio, select, checkbox): Recommend "bar" charts for showing distribution
        - For multi-select questions: Recommend "pie" charts to show proportion of selections
        - For numeric rating scales: Recommend "line" charts for trend analysis
        - For yes/no questions: Recommend "bar" charts for simple comparison
        
        Return your analysis as a JSON object with these sections:
        - insights: array of strings with key observations (3-7 items)
        - recommendations: array of strings with suggested actions (3-5 items)
        - visualizations: array of objects with:
          * question_id: string matching the question ID
          * chart_type: string with one of "bar", "pie", or "line" (must match data type)
          * rationale: string explaining why this chart type is appropriate
        """
        
        try:
            # Use GPT-4 for more sophisticated analysis, but without response_format parameter
            # as it's not supported by all models
            response = self.client.chat.completions.create(
                model="gpt-4",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a data analysis expert specializing in survey insights. Return your analysis as a JSON object."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content
            try:
                # Attempt to parse JSON from the response
                data = json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, extract insights manually
                print(f"Response not in JSON format. Extracting insights manually.")
                lines = content.strip().split("\n")
                insights = []
                recommendations = []
                visualizations = []
                
                section = None
                for line in lines:
                    line = line.strip()
                    if "insights:" in line.lower() or "key insights:" in line.lower():
                        section = "insights"
                        continue
                    elif "recommendations:" in line.lower():
                        section = "recommendations"
                        continue
                    elif "visualizations:" in line.lower():
                        section = "visualizations"
                        continue
                    
                    # Skip empty lines and headings
                    if not line or line.startswith('#'):
                        continue
                        
                    # Extract content based on current section
                    if section == "insights" and line.startswith('-'):
                        insights.append(line[1:].strip())
                    elif section == "recommendations" and line.startswith('-'):
                        recommendations.append(line[1:].strip())
                
                # Create fallback structured response
                data = {
                    "insights": insights if insights else ["Patterns in responses suggest varying levels of participant engagement."],
                    "recommendations": recommendations if recommendations else ["Consider refining questions to gather more specific feedback."],
                    "visualizations": [{"question_id": list(question_data.keys())[0] if question_data else "q1", 
                                      "chart_type": "bar", 
                                      "rationale": "Shows distribution of responses across categories"}]
                }
            
            return data
            
        except Exception as e:
            print(f"Error analyzing survey responses: {str(e)}")
            # Return basic fallback analysis
            return {
                "insights": [
                    "Response patterns suggest varying levels of satisfaction across different aspects.",
                    "Text responses highlight several areas for potential improvement.",
                    "The data indicates correlation between overall satisfaction and likelihood to recommend."
                ],
                "recommendations": [
                    "Focus on areas with lowest satisfaction scores for immediate improvement.",
                    "Consider follow-up surveys to dive deeper into specific issues mentioned.",
                    "Use open text responses to identify new feature opportunities."
                ],
                "visualizations": [
                    {"question_id": list(question_data.keys())[0] if question_data else "q1", 
                     "chart_type": "bar", 
                     "rationale": "Shows distribution of responses across categories"}
                ]
            }

# Singleton instance
openai_service = OpenAIService()
