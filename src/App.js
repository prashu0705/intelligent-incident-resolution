import React, { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const askAssistant = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/ask-assistant", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-session-id": "test-session-123"  // You can make this dynamic if needed
        },
        body: JSON.stringify({
          question: question,
          conversation_history: [],
          top_k: 3
        })
      });
      const data = await response.json();
      setAnswer(data.answer);
    } catch (error) {
      setAnswer("Error: " + error.message);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>L1 Agent Assistant Chat</h1>
      <textarea
        rows={4}
        cols={50}
        placeholder="Ask your question here..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />
      <br />
      <button onClick={askAssistant} disabled={loading || !question.trim()}>
        {loading ? "Loading..." : "Ask Assistant"}
      </button>
      <div style={{ marginTop: 20 }}>
        <strong>Answer:</strong>
        <p>{answer}</p>
      </div>
    </div>
  );
}

export default App;

