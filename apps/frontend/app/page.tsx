"use client";

import { useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";

const HEADER_HEIGHT_PX = 64;

// Types for game state
interface GameState {
  grid_width: number;
  grid_height: number;
  game_tick: number;
  snake: Array<[number, number]>; // Array of [x, y] coordinates
  food: [number, number]; // [x, y] coordinate
  score: number;
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<Socket | undefined>(undefined);

  // Variables for tracking the snake attributes
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [score, setScore] = useState<number>(0);
  const [isGameRunning, setIsGameRunning] = useState<boolean>(false);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [gamesPlayed, setGamesPlayed] = useState<number>(0);
  const [gridWidth, setGridWidth] = useState<number>(29);
  const [gridHeight, setGridHeight] = useState<number>(19);
  const [snakeBody, setSnakeBody] = useState<Array<[number, number]>>([]);
  const [foodPosition, setFoodPosition] = useState<[number, number]>([0, 0]);
  const [gameTick, setGameTick] = useState<number>(0.03);

  // Canvas dimensions and drawing constants
  const [canvasWidth, setCanvasWidth] = useState<number>(800);
  const [canvasHeight, setCanvasHeight] = useState<number>(600);
  const [cellSize, setCellSize] = useState<number>(20);

  // Calculate optimal canvas size based on grid dimensions
  const calculateCanvasSize = () => {
    const maxWidth = window.innerWidth * 0.8;
    const maxHeight = (window.innerHeight - HEADER_HEIGHT_PX) * 0.8;

    const cellWidth = Math.floor(maxWidth / gridWidth);
    const cellHeight = Math.floor(maxHeight / gridHeight);
    const optimalCellSize = Math.min(cellWidth, cellHeight, 25); // Max 25px per cell

    const newWidth = gridWidth * optimalCellSize;
    const newHeight = gridHeight * optimalCellSize;

    setCellSize(optimalCellSize);
    setCanvasWidth(newWidth);
    setCanvasHeight(newHeight);
  };

  // Function to draw the game to the canvas
  const drawGame = () => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");

    if (!context || !canvas) return;

    // Clear the canvas
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Get current theme colors
    const isDark = document.documentElement.classList.contains("dark");
    const backgroundColor = isDark ? "#0a0a0a" : "#ffffff";
    const gridColor = isDark ? "#262626" : "#e5e5e5";
    const snakeColor = isDark ? "#22c55e" : "#16a34a";
    const snakeHeadColor = isDark ? "#15803d" : "#166534";
    const foodColor = isDark ? "#ef4444" : "#dc2626";

    // Draw background
    context.fillStyle = backgroundColor;
    context.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid lines
    context.strokeStyle = gridColor;
    context.lineWidth = 1;

    // Vertical lines
    for (let x = 0; x <= gridWidth; x++) {
      context.beginPath();
      context.moveTo(x * cellSize, 0);
      context.lineTo(x * cellSize, canvas.height);
      context.stroke();
    }

    // Horizontal lines
    for (let y = 0; y <= gridHeight; y++) {
      context.beginPath();
      context.moveTo(0, y * cellSize);
      context.lineTo(canvas.width, y * cellSize);
      context.stroke();
    }

    // Draw snake
    if (snakeBody.length > 0) {
      snakeBody.forEach((segment, index) => {
        const [x, y] = segment;
        const isHead = index === 0;

        context.fillStyle = isHead ? snakeHeadColor : snakeColor;
        context.fillRect(
          x * cellSize + 1,
          y * cellSize + 1,
          cellSize - 2,
          cellSize - 2
        );

        // Add eyes to the head
        if (isHead) {
          context.fillStyle = backgroundColor;
          const eyeSize = cellSize * 0.15;
          const eyeOffset = cellSize * 0.25;

          // Left eye
          context.fillRect(
            x * cellSize + eyeOffset,
            y * cellSize + eyeOffset,
            eyeSize,
            eyeSize
          );

          // Right eye
          context.fillRect(
            x * cellSize + cellSize - eyeOffset - eyeSize,
            y * cellSize + eyeOffset,
            eyeSize,
            eyeSize
          );
        }
      });
    }

    // Draw food
    if (foodPosition) {
      const [x, y] = foodPosition;
      context.fillStyle = foodColor;

      // Draw food as a circle
      context.beginPath();
      context.arc(
        x * cellSize + cellSize / 2,
        y * cellSize + cellSize / 2,
        cellSize * 0.4,
        0,
        2 * Math.PI
      );
      context.fill();
    }
  };

  useEffect(() => {
    if (socketRef.current === undefined) {
      socketRef.current = io("localhost:8765");

      const onConnect = () => {
        console.log("Connected to server");
        socketRef.current?.emit("start_game", {
          grid_width: 29, // Grid width in cells (default: 29)
          grid_height: 19, // Grid height in cells (default: 19)
          starting_tick: 0.03, // Time between game updates in seconds
          use_ai: true, // Whether to use AI agent or manual control
        });
      };

      const onConnectionStatus = (data: any) => {
        console.log("Connection status:", data);
        setIsConnected(data.status === "connected");
      };

      const onGameStarted = (data: any) => {
        console.log("Game started:", data);
        setIsGameRunning(true);
        setGridWidth(data.grid_width);
        setGridHeight(data.grid_height);
        setGameTick(data.starting_tick);
      };

      const onGameState = (data: GameState) => {
        // Update all game state from server
        setGameState(data);
        setScore(data.score);
        setSnakeBody(data.snake);
        setFoodPosition(data.food);
        setGridWidth(data.grid_width);
        setGridHeight(data.grid_height);
        setGameTick(data.game_tick);
      };

      const onGameOver = (data: any) => {
        console.log("Game over:", data);
        setIsGameRunning(false);
        setScore(data.final_score);
        setGamesPlayed(data.games_played);
      };

      const onError = (data: any) => {
        console.error("Server error:", data.message);
      };

      const onWarning = (data: any) => {
        console.warn("Server warning:", data.message);
      };

      socketRef.current.on("connect", onConnect);
      socketRef.current.on("connection_status", onConnectionStatus);
      socketRef.current.on("game_started", onGameStarted);
      socketRef.current.on("game_state", onGameState);
      socketRef.current.on("game_over", onGameOver);
      socketRef.current.on("error", onError);
      socketRef.current.on("warning", onWarning);

      return () => {
        socketRef.current?.off("connect", onConnect);
        socketRef.current?.off("connection_status", onConnectionStatus);
        socketRef.current?.off("game_started", onGameStarted);
        socketRef.current?.off("game_state", onGameState);
        socketRef.current?.off("game_over", onGameOver);
        socketRef.current?.off("error", onError);
        socketRef.current?.off("warning", onWarning);
      };
    }
  }, []); // socket stuff

  // Initialize canvas size on mount
  useEffect(() => {
    calculateCanvasSize();
  }, [gridWidth, gridHeight]);

  // Draw the game whenever state changes
  useEffect(() => {
    drawGame();
  }, [gameState, snakeBody, foodPosition, canvasWidth, canvasHeight, cellSize]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");

    if (!context) {
      console.warn("Canvas 2D context is not available");
      return;
    }

    // Draw the game initially and on theme changes
    drawGame();

    const observer = new MutationObserver(() => {
      // Handle redrawing on theme change
      drawGame();
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => {
      observer.disconnect();
    };
  }, [drawGame]); // redraw

  useEffect(() => {
    const handleResize = () => {
      // Recalculate canvas size on window resize
      calculateCanvasSize();
    };

    // Initial calculation
    calculateCanvasSize();

    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [gridWidth, gridHeight]); // resize

  return (
    <div className="absolute top-16 left-0 right-0 bottom-0 flex flex-col items-center justify-center">
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        style={{ position: "absolute", border: "none", outline: "none" }}
      />

      {/* Game Info Overlay */}
      <div className="absolute top-4 left-4 right-4 flex justify-between items-start pointer-events-none">
        {/* Left side - Connection status */}
        <div className="bg-background/80 backdrop-blur-sm rounded-lg p-3 shadow-md">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? "bg-green-500" : "bg-red-500"
              }`}
            />
            <span className="text-sm font-medium">
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>

        {/* Right side - Game stats */}
        <div className="bg-background/80 backdrop-blur-sm rounded-lg p-3 shadow-md">
          <div className="text-right">
            <div className="text-2xl font-bold text-primary">
              Score: {score}
            </div>
            <div className="text-sm text-muted-foreground">
              Games: {gamesPlayed}
            </div>
            {isGameRunning && (
              <div className="text-xs text-green-600 dark:text-green-400">
                AI Playing...
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom overlay - Game info */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-background/80 backdrop-blur-sm rounded-lg p-4 shadow-md max-w-md">
        <div className="text-center">
          <h2 className="text-xl font-bold text-primary mb-2">
            Snake AI Training
          </h2>
          {!isGameRunning && !isConnected && (
            <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-2">
              Connecting to server...
            </p>
          )}
          {!isGameRunning && isConnected && (
            <p className="text-xs text-blue-600 dark:text-blue-400 mt-2">
              Starting new game...
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
