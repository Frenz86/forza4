from arena.game import Game
from arena.board import RED, YELLOW
from arena.llm import LLM
import gradio as gr
import pandas as pd


css = """
.dataframe-fix .table-wrap {
    min-height: 800px;
    max-height: 800px;
}
footer{display:none !important}
"""

js = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

ALL_MODEL_NAMES = LLM.all_model_names()


def message_html(game) -> str:
    """
    Return the message for the top of the UI
    """
    return f'<div style="text-align: center;font-size:18px">{game.board.message()}</div>'


def format_records_for_table(games):
    """
    Turn the results objects into a pandas DataFrame for the Gradio Dataframe
    """
    if not games:
        return pd.DataFrame(columns=["When", "Red Player", "Yellow Player", "Winner"])

    df = pd.DataFrame(
        [
            [
                game.when,
                game.red_player,
                game.yellow_player,
                "Red" if game.red_won else "Yellow" if game.yellow_won else "Draw",
            ]
            for game in reversed(games)
        ],
        columns=["When", "Red Player", "Yellow Player", "Winner"],
    )

    # Remove microseconds while preserving datetime format
    if not df.empty:
        df["When"] = pd.to_datetime(df["When"]).dt.floor("s")

    return df


def compute_win_stats(games):
    """
    Return a dict {player: (wins, total)} from the game history.
    Draws count as 0.5 wins for both players.
    """
    stats = {}
    for game in games:
        for player in [game.red_player, game.yellow_player]:
            if player not in stats:
                stats[player] = [0.0, 0]
        stats[game.red_player][1] += 1
        stats[game.yellow_player][1] += 1
        if game.red_won and not game.yellow_won:
            stats[game.red_player][0] += 1.0
        elif game.yellow_won and not game.red_won:
            stats[game.yellow_player][0] += 1.0
        else:
            stats[game.red_player][0] += 0.5
            stats[game.yellow_player][0] += 0.5
    return stats


def format_ratings_for_table(ratings, games):
    """
    Turn the ratings into a pandas DataFrame for the Gradio Dataframe,
    including win percentage.
    """
    win_stats = compute_win_stats(games)
    items = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    rows = []
    for player, elo in items:
        wins, total = win_stats.get(player, [0.0, 0])
        win_pct = f"{wins / total * 100:.1f}%" if total > 0 else "N/A"
        rows.append([player, int(round(elo)), total, win_pct])
    return pd.DataFrame(rows, columns=["Player", "ELO", "Games", "Win %"])


def load_callback(red_llm, yellow_llm):
    """
    Callback called when the game is started. Create a new Game object for the state.
    """
    game = Game(red_llm, yellow_llm)
    enabled = gr.Button(interactive=True)
    disabled = gr.Button(interactive=False)
    message = message_html(game)
    games = Game.get_games()
    records_df = format_records_for_table(games)
    ratings = Game.get_ratings()
    ratings_df = format_ratings_for_table(ratings, games)
    return (
        game,
        game.board.svg(),
        message,
        "",
        "",
        enabled,
        enabled,
        enabled,
        disabled,
        [False],
        records_df,
        ratings_df,
    )


def leaderboard_callback():
    """
    Callback called when the user switches to the Leaderboard tab. Load in the results.
    """
    games = Game.get_games()
    records_df = format_records_for_table(games)
    ratings = Game.get_ratings()
    ratings_df = format_ratings_for_table(ratings, games)
    return records_df, ratings_df


def move_callback(game):
    """
    Callback called when the user clicks to do a single move.
    """
    game.move()
    message = message_html(game)
    if_active = gr.Button(interactive=game.board.is_active())
    disabled = gr.Button(interactive=False)
    return (
        game,
        game.board.svg(),
        message,
        game.thoughts(RED),
        game.thoughts(YELLOW),
        if_active,
        if_active,
        disabled,
        [False],
    )


def run_callback(game, should_stop):
    """
    Callback called when the user runs an entire game. Reset the board, run the game, store results.
    Yield interim results so the UI updates.
    """
    enabled = gr.Button(interactive=True)
    disabled = gr.Button(interactive=False)
    stop_enabled = gr.Button(interactive=True)
    game.reset()
    message = message_html(game)
    yield (
        game,
        game.board.svg(),
        message,
        game.thoughts(RED),
        game.thoughts(YELLOW),
        disabled,
        disabled,
        disabled,
        stop_enabled,
        False,
    )
    while game.board.is_active():
        if should_stop[0]:
            yield (
                game,
                game.board.svg(),
                message,
                game.thoughts(RED),
                game.thoughts(YELLOW),
                enabled,
                enabled,
                enabled,
                disabled,
                True,
            )
            return
        game.move()
        message = message_html(game)
        yield (
            game,
            game.board.svg(),
            message,
            game.thoughts(RED),
            game.thoughts(YELLOW),
            disabled,
            disabled,
            disabled,
            stop_enabled,
            False,
        )
    game.record()
    yield (
        game,
        game.board.svg(),
        message,
        game.thoughts(RED),
        game.thoughts(YELLOW),
        disabled,
        disabled,
        enabled,
        disabled,
        False,
    )


def stop_callback(game):
    """
    Callback called when the user stops the game
    """
    enabled = gr.Button(interactive=True)
    disabled = gr.Button(interactive=False)
    return (
        game,
        game.board.svg(),
        message_html(game),
        game.thoughts(RED),
        game.thoughts(YELLOW),
        enabled,
        enabled,
        enabled,
        disabled,
        [True],
    )


def set_stop_flag(should_stop):
    """
    Set the stop flag to True
    """
    return [True]


def model_callback(player_name, game, new_model_name):
    """
    Callback when the user changes the model
    """
    player = game.players[player_name]
    player.switch_model(new_model_name)
    return game


def red_model_callback(game, new_model_name):
    """
    Callback when red model is changed
    """
    return model_callback(RED, game, new_model_name)


def yellow_model_callback(game, new_model_name):
    """
    Callback when yellow model is changed
    """
    return model_callback(YELLOW, game, new_model_name)


def player_section(name, default):
    """
    Create the left and right sections of the UI
    """
    with gr.Row():
        gr.HTML(f'<div style="text-align: center;font-size:18px">{name} Player</div>')
    with gr.Row():
        dropdown = gr.Dropdown(ALL_MODEL_NAMES, value=default, label="LLM", interactive=True)
    with gr.Row():
        gr.HTML('<div style="text-align: center;font-size:16px">Inner thoughts</div>')
    with gr.Row():
        thoughts = gr.HTML(label="Thoughts")
    return thoughts, dropdown


def make_display():
    """
    The Gradio UI to show the Game, with event handlers
    """
    with gr.Blocks(
        title="C4 Battle",
        css=css,
        js=js,
        theme=gr.themes.Default(primary_hue="sky"),
    ) as blocks:
        game = gr.State()
        should_stop = gr.State([False])

        with gr.Tabs():
            with gr.TabItem("Game"):
                with gr.Row():
                    gr.HTML(
                        '<div style="text-align: center;font-size:24px">Four-in-a-row LLM Showdown</div>'
                    )
                with gr.Row():
                    with gr.Column(scale=1):
                        red_thoughts, red_dropdown = player_section("Red", ALL_MODEL_NAMES[0])
                    with gr.Column(scale=2):
                        with gr.Row():
                            message = gr.HTML(
                                '<div style="text-align: center;font-size:18px">The Board</div>'
                            )
                        with gr.Row():
                            board_display = gr.HTML()
                        with gr.Row():
                            with gr.Column(scale=1):
                                move_button = gr.Button("Next move")
                            with gr.Column(scale=1):
                                run_button = gr.Button("Run game", variant="primary")
                            with gr.Column(scale=1):
                                stop_button = gr.Button("Stop", variant="stop")
                            with gr.Column(scale=1):
                                reset_button = gr.Button("Start Over")
                        with gr.Row():
                            gr.HTML(
                            )

                    with gr.Column(scale=1):
                        yellow_thoughts, yellow_dropdown = player_section(
                            "Yellow", ALL_MODEL_NAMES[1]
                        )
            with gr.TabItem("Leaderboard"):
                with gr.Row():
                    refresh_button = gr.Button("Refresh", variant="primary", scale=0)
                with gr.Row():
                    with gr.Column(scale=1):
                        ratings_df = gr.Dataframe(
                            headers=["Player", "ELO", "Games", "Win %"],
                            label="Ratings (recent models only)",
                            column_widths=[3, 1, 1, 1],
                            wrap=True,
                            interactive=False,
                            max_height=800,
                            elem_classes=["dataframe-fix"],
                        )
                    with gr.Column(scale=2):
                        results_df = gr.Dataframe(
                            headers=["When", "Red Player", "Yellow Player", "Winner"],
                            label="Game History",
                            column_widths=[2, 2, 2, 1],
                            wrap=True,
                            interactive=False,
                            max_height=800,
                            elem_classes=["dataframe-fix"],
                        )

        blocks.load(
            load_callback,
            inputs=[red_dropdown, yellow_dropdown],
            outputs=[
                game,
                board_display,
                message,
                red_thoughts,
                yellow_thoughts,
                move_button,
                run_button,
                reset_button,
                stop_button,
                should_stop,
                results_df,
                ratings_df,
            ],
        )
        move_button.click(
            move_callback,
            inputs=[game],
            outputs=[
                game,
                board_display,
                message,
                red_thoughts,
                yellow_thoughts,
                move_button,
                run_button,
                stop_button,
                should_stop,
            ],
        )
        red_dropdown.change(red_model_callback, inputs=[game, red_dropdown], outputs=[game])
        yellow_dropdown.change(
            yellow_model_callback, inputs=[game, yellow_dropdown], outputs=[game]
        )
        run_button.click(
            run_callback,
            inputs=[game, should_stop],
            outputs=[
                game,
                board_display,
                message,
                red_thoughts,
                yellow_thoughts,
                move_button,
                run_button,
                reset_button,
                stop_button,
                should_stop,
            ],
        )
        stop_button.click(
            set_stop_flag,
            inputs=[should_stop],
            outputs=[should_stop],
        )
        reset_button.click(
            load_callback,
            inputs=[red_dropdown, yellow_dropdown],
            outputs=[
                game,
                board_display,
                message,
                red_thoughts,
                yellow_thoughts,
                move_button,
                run_button,
                reset_button,
                stop_button,
                should_stop,
                results_df,
                ratings_df,
            ],
        )

        refresh_button.click(
            leaderboard_callback, inputs=[], outputs=[results_df, ratings_df]
        )

    return blocks
