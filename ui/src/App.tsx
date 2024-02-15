import { useEffect } from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import { Session, useStore } from "./store";
import { getSession, createSession } from "./api";
import { getValueFromUrlOrLocalstorage, writeToLocalStorage } from "./utils";
import InstructionsPage from "./pages/Instructions";
import TutorialStartPage from "./pages/Tutorial/Start";
import TutorialChoicePage from "./pages/Tutorial/Choice";
import TutorialCompletePage from "./pages/Tutorial/Complete";
import MainStartPage from "./pages/Main/Start";
import MainResolutionPage from "./pages/Main/Resolution";
import MainChoicePage from "./pages/Main/Choice";
import MainRoundCompletePage from "./pages/Main/RoundComplete";
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
    path: "/tutorial/choice",
    element: <TutorialChoicePage />,
  },
  {
    path: "/tutorial/complete",
    element: <TutorialCompletePage />,
  },
  {
    path: "/main/start",
    element: <MainStartPage />,
  },
  {
    path: "/main/resolution",
    element: <MainResolutionPage />,
  },
  {
    path: "/main/choice",
    element: <MainChoicePage />,
  },
  {
    path: "/main/round-complete",
    element: <MainRoundCompletePage />,
  },
  {
    path: "/complete",
    element: <ExperimentCompletePage />,
  },
]);

const App = () => {
  const setSession = useStore((state) => state.setSession);

  useEffect(() => {
    const setup = async () => {
      let sess;

      // check for existing session
      const sid = getValueFromUrlOrLocalstorage("sessionid");
      if (sid) {
        sess = (await getSession(sid)) as Session;
      }

      // create new session if none exists
      if (!sess) {
        sess = (await createSession(
          getValueFromUrlOrLocalstorage("expid"),
          getValueFromUrlOrLocalstorage("userid"),
          getValueFromUrlOrLocalstorage("texture"),
          Number(getValueFromUrlOrLocalstorage("cost")),
        )) as Session;
      }

      // save session id to localstorage and save session to global store
      writeToLocalStorage("sessionid", sess.id);
      setSession(sess);
    };
    setup();
  }, []);

  return (
    <div className="App">
      <RouterProvider router={router} />
    </div>
  );
};

export default App;
