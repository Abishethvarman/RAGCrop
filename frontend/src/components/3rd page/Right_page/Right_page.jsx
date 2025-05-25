import React, { useState, useEffect } from "react"; // Import useState, useEffect
import { useNavigate } from "react-router-dom"; // Import useNavigate
import axios from "axios"; // Import axios
import logo from "../../../assets/logo_.svg";
import { useLocation } from "react-router-dom";
import mini_logo from "../../../assets/mini_logo.svg";

const Right_Page = () => {
  const location = useLocation();
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const { result, question } = location.state || {}; // Handle potential undefined state

  const navigate = useNavigate(); // Initialize useNavigate

  const handleNavigate = () => {
    navigate("/");
  };

  return (
    <div className="w-[580px] h-[780px] bg-white rounded-[25px] p-10">
      <div>
        <img src={logo} alt="Main Logo" />
        <div className="border-t border-[#176B872A] mt-[31px]"></div>
      </div>

      <div className="bg-[#176B872A] mt-12 w-53 rounded-[12px] ml-72 ">
        <h1 className="text-[15px] text-[#04364A] px-5 py-2.5">{question}</h1>
      </div>

      <div className="flex mt-7.5">
        <div className="h-7.5 w-7.5 bg-[#176B87] flex items-center justify-center rounded-full">
          <img
            className="w-2.4 h-3.7 px-2.5 py-2"
            src={mini_logo}
            alt="Mini Logo"
          />
        </div>
        <textarea
          className="ml-3 p-3 w-full max-h-[700px] h-[600px] border-0 overflow-y-auto"
          value={result}
          readOnly
          style={{ minHeight: "150px", maxHeight: "400px", overflowY: "auto" }}
        ></textarea>
      </div>

      {/* Added LatestPrediction Component */}

      <div className="flex justify-center items-center mt-7">
        <button
          onClick={handleNavigate}
          className=" text-[15px] font-bold text-white bg-[#176B87] rounded-[100px] p-5 "
        >
          Start Fresh & Grow Again
        </button>
      </div>
    </div>
  );
};

export default Right_Page;
