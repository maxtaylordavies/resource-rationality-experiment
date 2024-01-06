import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { CountdownCircleTimer } from "react-countdown-circle-timer";

interface Props extends React.HTMLAttributes<HTMLDivElement> {
  duration: number;
  to: string;
}

export const CountdownLink = ({ duration, to, ...other }: Props) => {
  const navigate = useNavigate();
  const [timerKey, setTimerKey] = useState<number>(0);

  console.log(other);

  return (
    <div {...other}>
      <CountdownCircleTimer
        key={timerKey}
        isPlaying
        duration={duration}
        onComplete={(totalElapsedTime) => {
          setTimerKey(timerKey + 1);
          navigate(to);
        }}
        colors={"#3A3A3A"}
        trailColor="#d8d8d8"
        size={125}
      >
        {({ remainingTime }) => (
          <span style={{ fontSize: 36, fontWeight: 700, color: "#3A3A3A" }}>
            {remainingTime}
          </span>
        )}
      </CountdownCircleTimer>
    </div>
  );
};
