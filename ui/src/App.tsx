import React from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import InstructionsPage from "./pages/Instructions";
import TutorialStartPage from "./pages/Tutorial/Start";
import TutorialEvidencePage from "./pages/Tutorial/Evidence";
import TutorialTestPage from "./pages/Tutorial/Test";
import MainEvidencePage from "./pages/Main/Evidence";
import MainTestPage from "./pages/Main/Test";
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
    path: "/tutorial/test",
    element: <TutorialTestPage />,
  },
  {
    path: "/main/evidence",
    element: <MainEvidencePage />,
  },
  {
    path: "/main/test",
    element: <MainTestPage />,
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
