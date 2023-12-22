import React, { useEffect } from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import EvidencePage from "./pages/EvidencePage";
import TestPage from "./pages/TestPage";
import "./App.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <div>Hello world!</div>,
  },
  {
    path: "/evidence",
    element: <EvidencePage />,
  },
  {
    path: "/test",
    element: <TestPage />,
  },
]);

const App = () => {
  return (
    <div className="App">
      <RouterProvider router={router} />
    </div>
  );
};

export default App;
