import React, { useEffect } from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import { useStore } from "./store";
import { NUM_CHOICES } from "./constants";
import EvidencePage from "./pages/EvidencePage";
import TestPage from "./pages/TestPage";
import ExperimentCompletePage from "./pages/ExperimentCompletePage";
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
  {
    path: "/complete",
    element: <ExperimentCompletePage />,
  },
]);

const App = () => {
  const choiceCount = useStore((state) => state.choiceCount);

  useEffect(() => {
    if (choiceCount === NUM_CHOICES) {
      router.navigate("/complete");
    }
  }, [choiceCount]);

  return (
    <div className="App">
      <RouterProvider router={router} />
    </div>
  );
};

export default App;
