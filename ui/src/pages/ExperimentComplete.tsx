import { useState } from "react";

import { removeFromLocalStorage } from "../utils";
import { updateSession } from "../api";
import { useStore } from "../store";
import { Button } from "../components/Button/Button";
import { Box } from "../components/Box/Box";

const PROLIFIC_CODE = "C1LEZ1G5";

const ExperimentCompletePage = (): JSX.Element => {
  const session = useStore((state) => state.session);
  const score = useStore((state) => state.score);

  const [response, setResponse] = useState("");
  const [submitted, setSubmitted] = useState(true);

  const handleSubmit = async () => {
    if (session) {
      await updateSession(session.id, score, response);
      setSubmitted(true);
      removeFromLocalStorage("sessionid");
    }
  };

  const copyCode = async () => {
    await navigator.clipboard.writeText(PROLIFIC_CODE);
  };

  return (
    <Box className="page">
      {submitted ? (
        <Box
          justify="flex-start"
          align="flex-start"
          className="experiment-complete-container"
        >
          <h1>Thank you for participating!</h1>
          <p>
            Your Prolific completion code is: <strong>{PROLIFIC_CODE}</strong>.
            After copying this code, please close the tab.
          </p>
          <Button
            onClick={copyCode}
            label="Copy code"
            variant="primary"
            style={{ marginTop: "20px" }}
          />
        </Box>
      ) : (
        <Box
          justify="flex-start"
          align="flex-start"
          className="experiment-complete-container"
        >
          <h1>Almost finished</h1>
          <p>
            Please write a brief response (minimum 100 characters) to the
            following question: during the game, how did you make decisions
            about which map to purchase?
          </p>
          <textarea
            id="response"
            name="response"
            value={response}
            onChange={(e) => setResponse(e.target.value)}
            rows={5}
            // cols={50}
            placeholder="Your response here"
          />
          <Button
            onClick={handleSubmit}
            label="Submit"
            variant="primary"
            style={{
              marginTop: "20px",
              opacity: response.length < 100 ? 0.5 : 1,
              pointerEvents: response.length < 100 ? "none" : "auto",
            }}
          />
        </Box>
      )}
    </Box>
  );
};

export default ExperimentCompletePage;
