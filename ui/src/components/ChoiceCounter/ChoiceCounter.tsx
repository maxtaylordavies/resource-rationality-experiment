import React from "react";

import { useStore } from "../../store";
import "./choice-counter.css";

type Props = {
  phaseTitle: string;
  numChoices: number;
};

export const ChoiceCounter = ({
  numChoices,
  phaseTitle,
}: Props): JSX.Element => {
  const choiceCount = useStore((state) => state.choiceCount);

  return (
    <div className="choice-count-container">
      <span className="choice-count-phase-title">({phaseTitle})</span>
      <div className="choice-count-box">
        <span className="choice-count">
          {choiceCount}/{numChoices}
        </span>
        <span>choices</span>
      </div>
    </div>
  );
};
