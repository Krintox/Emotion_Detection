import React, { useState } from 'react';

function App() {
  const [inputText, setInputText] = useState('');
  const [prediction, setPrediction] = useState('');

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    // Make a POST request to the Flask API endpoint
    fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: inputText }),
    })
      .then((response) => response.json())
      .then((data) => {
        setPrediction(data.prediction);
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputText}
          onChange={handleInputChange}
          placeholder="Enter text"
        />
        <button type="submit">Predict</button>
      </form>
      {prediction && (
        <div>
          <h3>Prediction:</h3>
          <p>{prediction}</p>
        </div>
      )}
    </div>
  );
}

export default App;