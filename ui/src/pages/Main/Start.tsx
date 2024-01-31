import React from "react";

import { Box } from "../../components/Box/Box";
import { LinkButton } from "../../components/Button/LinkButton";

const StartPage = (): JSX.Element => {
  return (
    <Box className="page">
      <div className="tutorial-start-container">
        <h1>Main experiment</h1>
        <p>You will now play through the main experiment.</p>
        <LinkButton
          to="/main/evidence"
          label="Start"
          variant="primary"
          style={{ marginTop: 60 }}
        />
      </div>
    </Box>
  );
};

export default StartPage;
