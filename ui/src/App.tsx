import React, { useEffect } from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import { useStore } from "./store";
import { NUM_CHOICES } from "./constants";
import Instructions from "./pages/Instructions";
import Evidence1 from "./pages/Evidence1";
import Test1 from "./pages/Test1";
import Evidence2 from "./pages/Evidence2";
import Test2 from "./pages/Test2";
import ExperimentComplete from "./pages/ExperimentComplete";
import "./App.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Instructions />,
  },
  {
    path: "/evidence1",
    element: <Evidence1 />,
  },
  {
    path: "/test1",
    element: <Test1 />,
  },
  {
    path: "/evidence2",
    element: <Evidence2 />,
  },
  {
    path: "/test2",
    element: <Test2 />,
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
