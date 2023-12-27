import React, { useEffect } from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import { useStore } from "./store";
import { NUM_CHOICES } from "./constants";
import Instructions from "./pages/Instructions";
import EvidenceTraining from "./pages/EvidenceTraining";
import TestTraining from "./pages/TestTraining";
import EvidenceActual from "./pages/EvidenceActual";
import TestActual from "./pages/TestActual";
import ExperimentComplete from "./pages/ExperimentComplete";
import "./App.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Instructions />,
  },
  {
    path: "/evidence1",
    element: <EvidenceTraining />,
  },
  {
    path: "/test1",
    element: <TestTraining />,
  },
  {
    path: "/evidence2",
    element: <EvidenceActual />,
  },
  {
    path: "/test2",
    element: <TestActual />,
  },
  {
    path: "/complete",
    element: <ExperimentComplete />,
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
