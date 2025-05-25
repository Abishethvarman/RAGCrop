import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; // Import useNavigate
import back_btn from "../../../assets/back.svg";
import logo from "../../../assets/logo_.svg";
import send_btn from "../../../assets/send.svg";
import axios from "axios";

const prompts = [
  "Which crop suits my land?",
  "How can I improve my soil?",
  "What’s the best planting season?",
  "Which fertilizer should I use?",
  "What are the best farming techniques?",
];

const PageRight = () => {
  const [selectedPrompt, setSelectedPrompt] = useState("");
  const [userInput, setUserInput] = useState("");
  const navigate = useNavigate(); // Initialize useNavigate

  const handlePromptClick = (prompt) => {
    setSelectedPrompt(prompt);
    setUserInput(prompt);
  };

  const handleInputChange = (e) => {
    setUserInput(e.target.value);
    if (selectedPrompt) {
      setSelectedPrompt(""); 
    }
  };

  console.log(userInput);

  const handleSubmit = async (e) => {
    e.preventDefault();

    

   //if (validateFields()) {
    try {
      const response = await axios.post("http://127.0.0.1:5001/userQuery", { text: userInput });
    
      const last_prompt = response.data.prompt;
    console.log(last_prompt);
      const resultsResponse = await axios.post("http://127.0.0.1:5002/chat", { prompt: last_prompt });
    
      const results = resultsResponse.data.rag_response; // Assuming response contains data you need
    
      // Navigate with results data
      if (userInput.trim() !== "") {
        navigate("/results", { state: { result: results, question: userInput } });
      } else {
        alert("Please enter a question before proceeding.");
      }
    } catch (error) {
      console.error("Error sending data to backend:", error);
    }
   // }

    
  };

  const handleBack = () => {
    navigate("/");
  }

  return (
    <div className="w-[580px] h-[780px] bg-white rounded-[25px] p-6 flex flex-col">
      <div className="flex items-center">
        <img className="pl-[30px] pt-[33.55px]" src={back_btn} alt="Back" onClick={handleBack}/>
        <h1 className="pl-2 pt-7.5 text-cyan-950 text-[15px]">Edit inputs</h1>
      </div>

      <div className="flex flex-col items-center mt-4 flex-grow ">
        <img src={logo} alt="Logo" />
        <h1 className="text-[25px] font-bold text-cyan-950 mt-2">
          Get Answers, Grow Better
        </h1>
        <p className="text-[15px] text-[#04364A] opacity-50 max-w-[370px] text-center">
          Choose a default prompt or type your own question to get expert farming insights.
        </p>
      </div>

      <div className="mt-auto flex flex-col items-center w-full">
        <div className="flex flex-wrap justify-center gap-3 mt-[-500px]">
          {prompts.map((prompt) => (
            <button
              key={prompt}
              className={`px-4 py-2 rounded-full border text-[15px] ${
                selectedPrompt === prompt
                  ? "border-[#04364A] text-[#04364A] font-semibold"
                  : "border-[#04364A3A] text-[#04364A]"
              }`}
              onClick={() => handlePromptClick(prompt)}
            >
              {prompt}
            </button>
          ))}
        </div>

        <div
          className="flex items-center border rounded-[20px] px-4 py-2 w-full max-w-[500px] mx-auto transition-all duration-300 mt-[300px]"
          style={{ borderColor: userInput ? "#04364A" : "#04364A3A" }}
        >
          <input
            type="text"
            name="quan"
            placeholder="Type your prompt here..."
            value={userInput}
            onChange={handleInputChange}
            className="flex-1 bg-transparent outline-none text-[#04364A] text-[15px]"
          />
          <button onClick={handleSubmit} className="p-[12px] bg-cyan-950 w-10 h-10 rounded-full">
            <img className="flex items-center justify-center w-[15px] h-[15px]" src={send_btn} alt="Send" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default PageRight;