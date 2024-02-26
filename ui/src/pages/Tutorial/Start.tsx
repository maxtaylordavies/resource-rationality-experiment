import { Box } from "../../components/Box/Box";
import { LinkButton } from "../../components/Button/LinkButton";

const StartPage = (): JSX.Element => {
  return (
    <Box className="page">
      <Box
        justify="flex-start"
        align="flex-start"
        className="phase-start-container"
      >
        <h1>Tutorial</h1>
        <p>
          You will now play through a tutorial round, in order to familiarise
          yourself with how the game works.
        </p>
        <p>
          To makes things easier, the tutorial round uses a smaller field with
          only 9 plots (in the main game there will be 64).
        </p>
        <p>
          You can replay the tutorial round as many times as you want. Coins
          earned or lost will not contribute to your final score.
        </p>
        <LinkButton
          to="/tutorial/choice"
          label="Start tutorial"
          variant="primary"
          style={{ marginTop: 60 }}
        />
      </Box>
    </Box>
  );
};

export default StartPage;
