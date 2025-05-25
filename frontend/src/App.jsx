import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import NavBar from "./components/NavBar/NavBar";
import Hero_left from "./components/Hero_Left/Hero_left";
import Page_left from "./components/2nd page/Page_Left/Page_left";
import Left_page from "./components/3rd page/Left_page/Left_page";

const App = () => {
  return (
    <Router>
      <div>
        <NavBar />
        <Routes>
          {/* Hero Left contains Hero Right */}
          <Route path="/" element={<Hero_left />} />
          
          {/* Page Left contains Page Right */}
          <Route path="/ask-grownius" element={<Page_left />} />
          
          {/* Left Page contains Right Page */}
          <Route path="/results" element={<Left_page />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
