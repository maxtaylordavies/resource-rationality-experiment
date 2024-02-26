import { HTMLProps } from "react";

interface CoinProps extends HTMLProps<HTMLImageElement> {
  width?: number;
  height?: number;
}

export const Coin = ({ width, height, ...props }: CoinProps) => {
  return (
    <img
      width={width}
      height={height}
      src={window.location.origin + "/assets/coin.png"}
      alt="coin"
      {...props}
    />
  );
};
