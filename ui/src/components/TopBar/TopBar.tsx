import React from "react";

import { useStore } from "../../store";
import "./top-bar.css";

type Props = {
  numChoices: number;
};

export const TopBar = ({ numChoices }: Props): JSX.Element => {
  const choiceCount = useStore((state) => state.choiceCount);
  const score = useStore((state) => state.score);

  return (
    <div className="top-bar">
      <div />
      <div className="score-box">{score} points</div>
      <div className="choice-count-container">
        <div className="choice-count-box">
          <span className="choice-count">
            {choiceCount}/{numChoices}
          </span>
          <span>choices</span>
        </div>
      </div>
    </div>
  );
};
