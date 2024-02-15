import { useEffect } from "react";

import { removeFromLocalStorage } from "../utils";

const ExperimentCompletePage = (): JSX.Element => {
  useEffect(() => {
    removeFromLocalStorage("sessionid");
  }, []);

  return (
    <div className="page">
      <h1>Experiment Complete - thanks for participating!</h1>
    </div>
  );
};

export default ExperimentCompletePage;
