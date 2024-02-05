import { useNavigate } from "react-router-dom";

import { useStore } from "../../store";
import { Box } from "../../components/Box/Box";
import { Button } from "../../components/Button/Button";

const RoundCompletePage = (): JSX.Element => {
  const navigate = useNavigate();

  const [round, incrementRound] = useStore((state) => [
    state.round,
    state.incrementRound,
  ]);

  const handleClick = () => {
    incrementRound();
    navigate("/main/resolution");
  };

  return (
    <Box className="page">
      <div className="tutorial-start-container">
        <h1>Round {round} complete</h1>
        <Button
          label="Next round"
          variant="primary"
          style={{ marginTop: 60 }}
          onClick={handleClick}
        />
      </div>
    </Box>
  );
};

export default RoundCompletePage;
