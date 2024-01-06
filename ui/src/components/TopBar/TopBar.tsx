import React from "react";

import { useStore } from "../../store";
import "./top-bar.css";

type Props = {
  phase: string;
  numChoices: number;
};

export const TopBar = ({ phase, numChoices }: Props): JSX.Element => {
  const choiceCount = useStore((state) => state.choiceCount);
  const score = useStore((state) => state.score);

  return (
    <div className="top-bar">
      <div className="phase-name">{phase}</div>
      <div className="score-box">
        <span className="score-box-score">{score}</span> points
      </div>
      <div className="choice-count-box">
        <span className="choice-count">
          {choiceCount}/{numChoices}
        </span>
        <span>pairs done</span>
      </div>
    </div>
  );
};
