import {
  NUM_ROUNDS,
  NUM_CHOICES_TUTORIAL,
  NUM_CHOICES_MAIN,
  VEGETABLES,
} from "../../constants";
import { useStore } from "../../store";
import { Coin } from "../Coin/Coin";
import "./top-bar.css";

export const TopBar = (): JSX.Element => {
  const round = useStore((state) => state.round);
  const choiceCount = useStore((state) => state.choiceCount);
  const score = useStore((state) => state.score);

  const isTutorial = window.location.pathname.includes("tutorial");
  const roundText = isTutorial
    ? "Tutorial"
    : `Round ${round} of ${NUM_ROUNDS} ${VEGETABLES[round - 1]}`;
  const NUM_CHOICES = isTutorial ? NUM_CHOICES_TUTORIAL : NUM_CHOICES_MAIN;

  return (
    <div className="top-bar">
      <div className="round-text">{roundText}</div>
      <div className="score-box">
        <span className="score-box-score">{score}</span>
        <Coin height={28} />
      </div>
      <div className="choice-count-box">
        <span className="choice-count">
          {choiceCount}/{NUM_CHOICES}
        </span>
        <span>pairs done</span>
      </div>
    </div>
  );
};
