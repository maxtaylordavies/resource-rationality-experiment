import React from "react";
import { motion, HTMLMotionProps } from "framer-motion";

import "./button.css";

interface Props extends HTMLMotionProps<"button"> {
  label: string;
  onClick: () => void;
  variant: "primary" | "secondary";
  children?: React.ReactNode;
}

export const Button = ({
  label,
  onClick,
  variant,
  children,
  ...props
}: Props) => {
  return (
    <motion.button
      className={`button ${variant}`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      {...props}
    >
      {label} {children}
    </motion.button>
  );
};
