import React from "react";
import { useNavigate } from "react-router-dom";
import { MotionProps, motion } from "framer-motion";

import "./button.css";

// extend motion button props
interface Props extends MotionProps {
  label: string;
  to: string;
  variant: "primary" | "secondary";
}

export const LinkButton = ({ label, to, variant, ...props }: Props) => {
  const navigate = useNavigate();

  return (
    <motion.button
      className={`button ${variant}`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={() => navigate(to)}
      {...props}
    >
      {label}
    </motion.button>
  );
};
