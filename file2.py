from flask import Flask, render_template, request
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# Initialize the model and template
template = """
Answer the question below.

Here is the conversation history: {history}

Question: {question}

Answer: 
"""

model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

app = Flask(__name__)

# Store conversation history globally for now
conversation_history = ""

@app.route("/", methods=["GET", "POST"])
def home():
    global conversation_history
    if request.method == "POST":
        user_input = request.form["user_input"]
        if user_input.lower() == 'exit':
            return render_template("index.html", history=conversation_history, response="Goodbye!")
        result = chain.invoke({"history": conversation_history, "question": user_input})
        conversation_history += f"\nUser: {user_input}\nAI: {result}"
        return render_template("index.html", history=conversation_history, response=result)
    return render_template("index.html", history=conversation_history, response="")

if __name__ == "__main__":
    app.run(debug=True)
