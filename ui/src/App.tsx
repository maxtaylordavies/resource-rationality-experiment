import React from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import InstructionsPage from "./pages/Instructions";
import TutorialStartPage from "./pages/Tutorial/Start";
import TutorialEvidencePage from "./pages/Tutorial/Evidence";
import TutorialChoicePage from "./pages/Tutorial/Choice";
import TutorialCompletePage from "./pages/Tutorial/Complete";
import MainEvidencePage from "./pages/Main/Evidence";
import MainChoicePage from "./pages/Main/Choice";
import ExperimentCompletePage from "./pages/ExperimentComplete";
import "./App.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <InstructionsPage />,
  },
  {
    path: "/tutorial/start",
    element: <TutorialStartPage />,
  },
  {
    path: "/tutorial/evidence",
    element: <TutorialEvidencePage />,
  },
  {
    path: "/tutorial/choice",
    element: <TutorialChoicePage />,
  },
  {
    path: "/tutorial/complete",
    element: <TutorialCompletePage />,
  },
  {
    path: "/main/evidence",
    element: <MainEvidencePage />,
  },
  {
    path: "/main/choice",
    element: <MainChoicePage />,
  },
  {
    path: "/complete",
    element: <ExperimentCompletePage />,
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
