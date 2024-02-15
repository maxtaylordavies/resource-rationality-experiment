const PREFIX = "_rre_";

export const getValueFromUrlOrLocalstorage = (key: string) => {
  const params = new URLSearchParams(window.location.search);
  return (
    params.get(key) ||
    JSON.parse(localStorage.getItem(`${PREFIX}${key}`) || "null")
  );
};

export const getProlificMetadata = () => {
  const metadata: { [key: string]: string } = {};
  const params = new URLSearchParams(window.location.search);
  params.forEach((val, key) => {
    if (key.startsWith("PRLFC")) {
      metadata[key] = val;
    }
  });
  return metadata;
};

export const writeToLocalStorage = (key: string, val: any) => {
  localStorage.setItem(`${PREFIX}${key}`, JSON.stringify(val));
};

export const removeFromLocalStorage = (key: string) => {
  localStorage.removeItem(`${PREFIX}${key}`);
};
