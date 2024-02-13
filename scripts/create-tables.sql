CREATE TABLE IF NOT EXISTS football.players (
  id INT UNIQUE GENERATED ALWAYS AS IDENTITY, 
  name VARCHAR (100) NOT NULL,   
  year_turned_pro TIMESTAMP, 
  position VARCHAR(2),
  auxiliary_positions VARCHAR(15)
);

CREATE TABLE IF NOT EXISTS football.teams(
    id INT UNIQUE GENERATED ALWAYS AS IDENTITY,
    team_name VARCHAR (50)
);

CREATE TABLE IF NOT EXISTS football.player_plays_for_team(
    player_id INT NOT NULL,
    team_id INT NOT NULL,
    player_name VARCHAR(100),
    effective_date TIMESTAMP NOT NULL,
    limit_date TIMESTAMP,
    PRIMARY KEY(player_id, team_id, effective_date),
    CONSTRAINT fk_player_id
        FOREIGN KEY(player_id) 
            REFERENCES football.players(id),
    CONSTRAINT fk_team_id
        FOREIGN KEY(team_id)
            REFERENCES football.teams(id)
);

CREATE TABLE IF NOT EXISTS football.player_rushing_stats(
    year TIMESTAMP NOT NULL,
    player_id INT NOT NULL,
    attempts INT,
    yards INT,
    tds INT,
    first_downs_rushing INT,
    rush_success_rate NUMERIC(4,1),
    longest_rush INT,
    yards_per_attempt NUMERIC(3,1),
    yards_per_game NUMERIC(4,1),
    fumbles INT,
    CONSTRAINT fk_player_id
        FOREIGN KEY(player_id) 
            REFERENCES football.players(id)
);

CREATE TABLE IF NOT EXISTS football.player_receiving_stats(
    year TIMESTAMP NOT NULL,
    player_id INT NOT NULL,
    targets INT,
    receptions INT,
    catch_per NUMERIC(4,1),
    yards INT,
    yards_per_rec NUMERIC(3,1),
    tds INT,
    first_downs_receiving INT,
    receiving_success_rate NUMERIC(4,1),
    longest_reception INT,
    yards_per_target NUMERIC(3,1),
    receptions_per_game NUMERIC(3,1),
    yards_per_game NUMERIC(4,1),
    fumbles INT,
    PRIMARY KEY(year, player_id),
    CONSTRAINT fk_player_id
        FOREIGN KEY(player_id) 
            REFERENCES football.players(id)
);

CREATE TABLE IF NOT EXISTS football.player_passing_stats(
    year TIMESTAMP NOT NULL,
    player_id INT NOT NULL,
    completions INT,
    attempts INT,
    completion_pec NUMERIC(3,1),
    yards INT,
    yards_per_attempt NUMERIC(3,1),
    adjusted_yards_per_attempt NUMERIC(3,1),
    tds INT,
    td_pec NUMERIC(3,1),
    pass_efficiency_rating NUMERIC(4,1),
    interceptions INT,
    interceptions_pec NUMERIC(3,1),
    first_downs_passing INT,
    pass_success_rate NUMERIC(4,1),
    longest_pass INT,
    rating NUMERIC(4,1),
    qbr NUMERIC(4,1),
    PRIMARY KEY(year, player_id),
    CONSTRAINT fk_player_id
        FOREIGN KEY(player_id) 
            REFERENCES football.players(id)
);

CREATE TABLE IF NOT EXISTS football.player_season(
    year TIMESTAMP NOT NULL,
    player_id INT NOT NULL,
    player_age INT,
    games_played INT,
    games_started INT,
    is_college_season BOOLEAN,
    PRIMARY KEY(year, player_id),
    CONSTRAINT fk_player_id
       FOREIGN KEY(player_id) 
           REFERENCES football.players(id)
);

alter table football.players
    add cfdb_player_id INT,
    add height INT,
    add weight INT;

alter table football.player_receiving_stats 
    add overall_usage NUMERIC (5,4),
    add pass_usage NUMERIC (5,4),
    add rush_usage NUMERIC (5,4),
    add first_down_usage NUMERIC (5,4),
    add second_down_usage NUMERIC (5,4),
    add third_down_usage NUMERIC (5,4),
    add standard_downs_usage NUMERIC (5,4),
    add passing_downs_usage NUMERIC (5,4),
    add average_ppa_all NUMERIC (4,3),
    add average_ppa_pass NUMERIC (4,3),
    add average_ppa_rush NUMERIC (4,3),
    add average_ppa_first_down NUMERIC (4,3),
    add average_ppa_second_down NUMERIC (4,3),
    add average_ppa_third_down NUMERIC (4,3),
    add average_ppa_standard_downs NUMERIC (4,3),
    add average_ppa_passing_down NUMERIC (4,3),
    add total_ppa_all NUMERIC (6,3),
    add total_ppa_pass NUMERIC (6,3),
    add total_ppa_rush NUMERIC (6,3),
    add total_ppa_first_down NUMERIC (6,3),
    add total_ppa_second_down NUMERIC (6,3),
    add total_ppa_third_down NUMERIC (6,3),
    add total_ppa_standard_downs NUMERIC (6,3),
    add total_ppa_passing_down NUMERIC (6,3);
    
alter table football.player_receiving_stats 
    drop overall_usage,
    drop pass_usage,
    drop rush_usage,
    drop first_down_usage,
    drop second_down_usage,
    drop third_down_usage,
    drop standard_downs_usage,
    drop passing_downs_usage,
    drop average_ppa_all,
    drop average_ppa_pass,
    drop average_ppa_rush,
    drop average_ppa_first_down,
    drop average_ppa_second_down,
    drop average_ppa_third_down,
    drop average_ppa_standard_downs,
    drop average_ppa_passing_down,
    drop total_ppa_all,
    drop total_ppa_pass,
    drop total_ppa_rush,
    drop total_ppa_first_down,
    drop total_ppa_second_down,
    drop total_ppa_third_down,
    drop total_ppa_standard_downs,
    drop total_ppa_passing_down;


alter table football.player_season
    add cfdb_season_added BOOLEAN;


alter table football.players
    alter column position type VARCHAR(5);

alter table football.player_receiving_stats
    alter column receiving_success_rate NUMERIC(4,1);