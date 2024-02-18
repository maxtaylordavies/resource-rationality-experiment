import React, { useEffect } from "react";

import { useStore } from "../../store";
import { Box } from "../../components/Box/Box";
import { LinkButton } from "../../components/Button/LinkButton";

const CompletePage = (): JSX.Element => {
  const resetChoiceCount = useStore((state) => state.resetChoiceCount);
  const resetScore = useStore((state) => state.resetScore);

  useEffect(() => {
    resetChoiceCount();
    resetScore();
  }, []);

  return (
    <Box className="page">
      <div className="tutorial-complete-container">
        <h1>Tutorial complete ✅</h1>
        <p>
          Well done! You can now proceed to the main experiment - or if you
          want, you can replay the tutorial round.
        </p>
        <div className="button-container">
          <LinkButton
            to="/tutorial/start"
            label="Replay tutorial"
            variant="secondary"
            style={{ marginRight: 10 }}
          />
          <LinkButton to="/main/start" label="Proceed" variant="primary" />
        </div>
      </div>
    </Box>
  );
};

export default CompletePage;
