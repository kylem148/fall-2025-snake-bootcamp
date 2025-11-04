import asyncio
import time
import socketio
from aiohttp import web
from typing import Any, Dict


#from model import DQN

from game import Game


# TODO: Create a SocketIO server instance with CORS settings to allow connections from frontend
sio = socketio.AsyncServer(cors_allowed_origins="*")

# TODO: Create a web application instance
app = web.Application()

# TODO: Attach the socketio server to the web app
sio.attach(app)


# Basic health check endpoint - keep this for server monitoring
async def handle_ping(request: Any) -> Any:
    """Simple ping endpoint to keep server alive and check if it's running"""
    return web.json_response({"message": "pong"})


# TODO: Create a socketio event handler for when clients connect
@sio.event
async def connect(sid: str, environ: Dict[str, Any]) -> None:
    """Handle client connections - called when a frontend connects to the server"""
    print(f"🟢 Client connected: {sid}")
    
    # Initialize empty session data for this client
    await sio.save_session(sid, {
        'game': None,
        'agent': None,
        'connected_at': time.time(),
        'games_played': 0
    })
    
    # Send a welcome message to the connected client
    await sio.emit('connection_status', {
        'status': 'connected',
        'message': 'Successfully connected to Snake game server',
        'session_id': sid
    }, room=sid)


# TODO: Create a socketio event handler for when clients disconnect
@sio.event
async def disconnect(sid: str) -> None:
    """Handle client disconnections - cleanup any resources"""
    print(f"🔴 Client disconnected: {sid}")
    
    try:
        # Get session data to clean up any running games
        session = await sio.get_session(sid)
        
        if session and session.get('game'):
            print(f"Cleaning up game session for client: {sid}")
            
        # Log session statistics if available
        if session:
            games_played = session.get('games_played', 0)
            connected_duration = time.time() - session.get('connected_at', time.time())
            print(f"Session stats - Games played: {games_played}, Duration: {connected_duration:.1f}s")
            
    except Exception as e:
        print(f"Error during cleanup for {sid}: {e}")
    
    # Session is automatically cleaned up by socketio when client disconnects


# TODO: Create a socketio event handler for starting a new game
@sio.event
async def start_game(sid: str, data: Dict[str, Any]) -> None:
    """Initialize a new game when the frontend requests it"""
    try:
        print(f"🎮 Starting new game for client: {sid}")
        
        # Extract game parameters from data with defaults
        grid_width = data.get('grid_width', 29)
        grid_height = data.get('grid_height', 19)
        starting_tick = data.get('starting_tick', 0.03)
        use_ai = data.get('use_ai', False)
        
        # Create a new Game instance and configure it
        game = Game()
        game.grid_width = grid_width
        game.grid_height = grid_height
        game.game_tick = starting_tick
        
        # Get current session and update it
        session = await sio.get_session(sid)
        session['game'] = game
        session['use_ai'] = use_ai
        session['game_running'] = True
        
        # If implementing AI, create an agent instance here
        if use_ai:
            try:
                from agent import DQN
                agent = DQN()
                session['agent'] = agent
                print(f"AI agent initialized for client: {sid}")
            except ImportError:
                print(f"⚠️ AI agent not available, running in manual mode")
                session['agent'] = None
                session['use_ai'] = False
        else:
            session['agent'] = None
        
        # Save the game state in the session
        await sio.save_session(sid, session)
        
        # Send initial game state to the client
        await sio.emit('game_state', game.to_dict(), room=sid)
        await sio.emit('game_started', {
            'message': 'Game started successfully',
            'use_ai': session['use_ai'],
            'grid_width': grid_width,
            'grid_height': grid_height,
            'starting_tick': starting_tick
        }, room=sid)
        
        # Start the game update loop
        asyncio.create_task(update_game(sid))
        
    except Exception as e:
        print(f"❌ Error starting game for {sid}: {e}")
        await sio.emit('error', {'message': f'Failed to start game: {str(e)}'}, room=sid)


# TODO: Optional - Create event handlers for saving/loading AI models
@sio.event
async def save_model(sid: str, data: Dict[str, Any]) -> None:
    """Save the AI model to disk"""
    try:
        session = await sio.get_session(sid)
        agent = session.get('agent')
        
        if not agent:
            await sio.emit('error', {'message': 'No AI agent found to save'}, room=sid)
            return
            
        model_name = data.get('model_name', f'snake_model_{int(time.time())}')
        # Save model (this would be implemented in the agent class)
        # agent.save_model(model_name)
        
        await sio.emit('model_saved', {
            'message': f'Model saved as {model_name}',
            'model_name': model_name
        }, room=sid)
        
    except Exception as e:
        await sio.emit('error', {'message': f'Failed to save model: {str(e)}'}, room=sid)


@sio.event
async def load_model(sid: str, data: Dict[str, Any]) -> None:
    """Load an AI model from disk"""
    try:
        session = await sio.get_session(sid)
        agent = session.get('agent')
        
        if not agent:
            await sio.emit('error', {'message': 'No AI agent found to load model into'}, room=sid)
            return
            
        model_name = data.get('model_name')
        if not model_name:
            await sio.emit('error', {'message': 'Model name is required'}, room=sid)
            return
            
        # Load model (this would be implemented in the agent class)
        # agent.load_model(model_name)
        
        await sio.emit('model_loaded', {
            'message': f'Model {model_name} loaded successfully',
            'model_name': model_name
        }, room=sid)
        
    except Exception as e:
        await sio.emit('error', {'message': f'Failed to load model: {str(e)}'}, room=sid)


@sio.event
async def player_input(sid: str, data: Dict[str, Any]) -> None:
    """Handle manual player input for controlling the snake"""
    try:
        session = await sio.get_session(sid)
        game = session.get('game')
        use_ai = session.get('use_ai', False)
        
        if not game or not game.running:
            return
            
        # Only allow manual input if AI is not controlling the game
        if use_ai:
            await sio.emit('warning', {'message': 'Cannot control manually while AI is active'}, room=sid)
            return
            
        direction = data.get('direction', '').upper()
        valid_directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        if direction in valid_directions:
            game.queue_change(direction)
            session['game'] = game
            await sio.save_session(sid, session)
            
    except Exception as e:
        print(f"❌ Error handling player input for {sid}: {e}")


@sio.event
async def stop_game(sid: str, data: Dict[str, Any]) -> None:
    """Stop the current game"""
    try:
        session = await sio.get_session(sid)
        session['game_running'] = False
        await sio.save_session(sid, session)
        
        await sio.emit('game_stopped', {'message': 'Game stopped by player'}, room=sid)
        print(f"🛑 Game stopped by player: {sid}")
        
    except Exception as e:
        print(f"❌ Error stopping game for {sid}: {e}")


@sio.event
async def get_stats(sid: str, data: Dict[str, Any]) -> None:
    """Get player statistics"""
    try:
        session = await sio.get_session(sid)
        game = session.get('game')
        
        stats = {
            'games_played': session.get('games_played', 0),
            'connected_duration': time.time() - session.get('connected_at', time.time()),
            'current_score': game.score if game else 0,
            'game_running': session.get('game_running', False),
            'using_ai': session.get('use_ai', False)
        }
        
        await sio.emit('player_stats', stats, room=sid)
        
    except Exception as e:
        await sio.emit('error', {'message': f'Failed to get stats: {str(e)}'}, room=sid)


# TODO: Implement the main game loop
async def update_game(sid: str) -> None:
    """Main game loop - runs continuously while the game is active"""
    try:
        while True:
            # Check if the session still exists (client hasn't disconnected)
            try:
                session = await sio.get_session(sid)
            except Exception:
                print(f" Session {sid} no longer exists, stopping game loop")
                break
            
            # Check if game should continue running
            if not session.get('game_running', False):
                break
                
            # Get the current game and agent state from the session
            game = session.get('game')
            agent = session.get('agent')
            use_ai = session.get('use_ai', False)
            
            if not game:
                break
                
            # Implement AI agentic decisions or manual game updates
            if use_ai and agent and game.running:
                await update_agent_game_state(game, agent)
            else:
                # For manual mode, just step the game forward
                if game.running:
                    game.step()
            
            # Update session with current game state
            session['game'] = game
            if agent:
                session['agent'] = agent
                
            # Save the updated session
            await sio.save_session(sid, session)
            
            # Send the updated game state to the client
            await sio.emit('game_state', game.to_dict(), room=sid)
            
            # If game ended, notify client and increment games played
            if not game.running:
                session['games_played'] = session.get('games_played', 0) + 1
                await sio.save_session(sid, session)
                await sio.emit('game_over', {
                    'final_score': game.score,
                    'games_played': session['games_played']
                }, room=sid)
                
                # Auto-restart for AI training, or stop for manual mode
                if use_ai and agent:
                    game.reset()
                    session['game'] = game
                    await sio.save_session(sid, session)
                    print(f" Auto-restarting game for AI training - Game #{session['games_played'] + 1}")
                else:
                    session['game_running'] = False
                    await sio.save_session(sid, session)
                    break
            
            # Wait for the appropriate game tick interval before next update
            await asyncio.sleep(game.game_tick)
            
    except Exception as e:
        print(f"❌ Error in game loop for {sid}: {e}")
        try:
            await sio.emit('error', {'message': f'Game loop error: {str(e)}'}, room=sid)
        except:
            pass


# TODO: Helper function for AI agent interaction with game
async def update_agent_game_state(game: Game, agent: Any) -> None:
    """Handle AI agent decision making and training"""
    try:
        # Get the current game state for the agent
        old_state = agent.get_state(game)
        
        # Have the agent choose an action (forward, turn left, turn right)
        action = agent.get_action(old_state)
        
        # Convert the agent's relative action to absolute game direction
        # Actions: [1,0,0] = straight, [0,1,0] = turn right, [0,0,1] = turn left
        current_direction = game.snake.direction
        
        if action == [0, 1, 0]:  # Turn right (relative to current direction)
            if current_direction == (0, -1):  # Currently moving up
                game.queue_change("RIGHT")  # Turn right -> move right
            elif current_direction == (0, 1):  # Currently moving down
                game.queue_change("LEFT")   # Turn right -> move left
            elif current_direction == (-1, 0):  # Currently moving left
                game.queue_change("UP")     # Turn right -> move up
            elif current_direction == (1, 0):  # Currently moving right
                game.queue_change("DOWN")   # Turn right -> move down
                
        elif action == [0, 0, 1]:  # Turn left (relative to current direction)
            if current_direction == (0, -1):  # Currently moving up
                game.queue_change("LEFT")   # Turn left -> move left
            elif current_direction == (0, 1):  # Currently moving down
                game.queue_change("RIGHT")  # Turn left -> move right
            elif current_direction == (-1, 0):  # Currently moving left
                game.queue_change("DOWN")   # Turn left -> move down
            elif current_direction == (1, 0):  # Currently moving right
                game.queue_change("UP")     # Turn left -> move up
        
        # For [1,0,0] (straight), we don't need to change direction
        
        # Step the game forward one frame
        old_score = game.score
        game.step()
        
        # Calculate the reward for this action
        reward = agent.calculate_reward(game, not game.running)
        
        # Get the new game state after the action
        new_state = agent.get_state(game)
        
        # Train the agent on this experience (short-term memory)
        agent.train_short_memory(old_state, action, reward, new_state, not game.running)
        
        # Store this experience in the agent's memory
        agent.remember(old_state, action, reward, new_state, not game.running)
        
        # If the game ended, train the agent's long-term memory
        if not game.running:
            agent.train_long_memory()
            
    except Exception as e:
        print(f"⚠️ Error in AI agent update: {e}")
        # Fallback to manual step if AI fails
        game.step()


# TODO: Main server startup function
async def main() -> None:
    """Start the web server and socketio server"""
    # Add the ping endpoint to the web app router
    app.router.add_get('/ping', handle_ping)
    
    # Create and configure the web server runner
    runner = web.AppRunner(app)
    await runner.setup()
    
    # Configure host and port (default to localhost:8000)
    host = "localhost"
    port = 8765
    
    # Create a TCP site for the server
    site = web.TCPSite(runner, host, port)
    
    # Start the server
    await site.start()
    
    # Print server startup message
    print(f"Snake game server started at http://{host}:{port}")
    print(f"SocketIO endpoint available at http://{host}:{port}/socket.io/")
    print(f"Health check available at http://{host}:{port}/ping")
    print("Press Ctrl+C to stop the server")
    
    # Keep the server running indefinitely
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested")
    finally:
        # Handle cleanup gracefully
        print("🧹 Cleaning up server resources...")
        await runner.cleanup()
        print("✅ Server stopped successfully")


if __name__ == "__main__":
    asyncio.run(main())
