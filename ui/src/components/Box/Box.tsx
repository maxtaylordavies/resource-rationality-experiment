import React from "react";
import { motion, HTMLMotionProps } from "framer-motion";

export interface Props extends HTMLMotionProps<"div"> {
  children: React.ReactNode;
  duration?: number;
}

export const Box = ({
  children,
  duration = 0.6,
  ...props
}: Props): JSX.Element => {
  return (
    <motion.div
      {...props}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration }}
      exit={{ opacity: 0 }}
    >
      {children}
    </motion.div>
  );
};
