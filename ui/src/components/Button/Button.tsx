import React from "react";
import { motion, MotionProps } from "framer-motion";

import "./button.css";

interface Props extends MotionProps {
  label: string;
  onClick: () => void;
  variant: "primary" | "secondary";
}

export const Button = ({ label, onClick, variant, ...props }: Props) => {
  return (
    <motion.button
      className={`button ${variant}`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      {...props}
    >
      {label}
    </motion.button>
  );
};
