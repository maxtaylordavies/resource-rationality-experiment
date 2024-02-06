import { useEffect } from "react";

import { useStore } from "../../store";
import { Box } from "../../components/Box/Box";
import { LinkButton } from "../../components/Button/LinkButton";

const StartPage = (): JSX.Element => {
  const incrementRound = useStore((state) => state.incrementRound);

  useEffect(() => {
    incrementRound();
  }, []);

  return (
    <Box className="page">
      <div className="phase-start-container">
        <h1>Main experiment</h1>
        <p>You will now play through the main experiment.</p>
        <p>
          The main difference from the tutorial is that you will have to{" "}
          <b>buy</b> a map at each round, rather than having it provided.
        </p>
        <p>
          You can choose between different <b>resolutions</b> of map - a more
          detailed map will cost more coins.
        </p>
        <p>
          Regardless of which map you buy, the actual field will always have 64
          plots.
        </p>
        <LinkButton
          to="/main/resolution"
          label="Start"
          variant="primary"
          style={{ marginTop: 60 }}
        />
      </div>
    </Box>
  );
};

export default StartPage;
