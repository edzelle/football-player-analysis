using FootballDataReader.Client;
using FootballDataReader.Data;
using FootballDataReader.Data.Entities;
using FootballDataReader.Logic.IService;
using FootballDataReader.Models;
using FootballDataReader.Models.CFB;
using FootballDataReader.Models.Constants;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Microsoft.VisualBasic.FileIO;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

namespace FootballDataReader.Logic.Service
{
    public class FootballService : IFootballService
    {
        private readonly FootballContext _context;
        private readonly CollegeFootballHttpClient _client;
        private readonly ILogger<IFootballService> _logger;

        public FootballService(FootballContext context, CollegeFootballHttpClient client, ILogger<IFootballService> logger)
        {
            _context = context;
            _client = client;
            _logger = logger;
        }

        public async Task ProcessRookieReceiverDataAndSaveToFile(string path)
        {
            var files = Directory.GetFiles(path, "*.csv").Reverse().ToList();
            foreach (string file in files)
            {
                if (!File.Exists(file))
                {
                    throw new Exception("File does not exist: " + file);
                }
                var year = file.Split('\\').Last().Substring(0, 4);
                if (int.Parse(year) > 2012)
                {
                    continue;
                }
                using (TextFieldParser csvParser = new TextFieldParser(file))
                {
                    csvParser.SetDelimiters(new string[] { "," });
                    csvParser.HasFieldsEnclosedInQuotes = false;
                    csvParser.ReadLine();

                    while (!csvParser.EndOfData)
                    {
                        string[] fields = csvParser.ReadFields();
                        string Name = fields[1].Trim('*');
                        string TeamName = fields[2];
                        string Games = fields[4];
                        string Receptions = fields[5];
                        string Yards = fields[6];
                        string YardsPerReception = fields[7];
                        string TDs = fields[8];
                        string pfrId = fields[17];

                        var player = await _context.Players.Where(x => x.Name == Name).FirstOrDefaultAsync();

                                        
                        if (player == null)
                        {
                            try
                            {
                                player = await CreateNewCollegePlayer(Name, Games, year, pfrId);
                            } catch (Exception ex)
                            {
                                throw;
                            }
                        }
                        else
                        {
                            if (player.ProFootballReferenceId != null && player.ProFootballReferenceId != pfrId)
                            {
                                player = await CreateNewCollegePlayer(Name, Games, year, pfrId);
                            }
                            else
                            {
                                var priorPlayerSeason = await _context.PlayerSeason.Where(x => x.PlayerId == player.Id).OrderBy(x => x.Year).ToListAsync();
                                var lastDocumentedSeason = priorPlayerSeason.FirstOrDefault();
                                if (lastDocumentedSeason != null && lastDocumentedSeason.Year.Year - int.Parse(year) != 1 && lastDocumentedSeason.Year.Year - int.Parse(year) != 0)
                                {
                                    
                                    // New Player with same name
                                    player = await CreateNewCollegePlayer(Name, Games, year, pfrId);
                                    
                                }
                            }
                        }

                        var team = await _context.Teams.Where(x => x.TeamName == TeamName).FirstOrDefaultAsync();
                        if (team == null)
                        {
                            team = await CreateNewTeam(TeamName);
                        }

                        var playerReceivingStats = await _context.PlayerReceivingStats.Where(x => x.PlayerId == player.Id && x.Year == new DateTime(int.Parse(year), 1, 1)).FirstOrDefaultAsync();
                        if (playerReceivingStats == null)
                        {
                            playerReceivingStats = new PlayerReceivingStats()
                            {
                                PlayerId = player.Id,
                                TDs = int.Parse(TDs),
                                Receptions = int.Parse(Receptions),
                                Yards = int.Parse(Yards),
                                YardsPerReception = int.Parse(Yards) / int.Parse(Receptions),
                                Year = new DateTime(int.Parse(year), 1, 1),
                            };
                            _context.Add(playerReceivingStats);
                        }


                        var playerSeason = await _context.PlayerSeason.Where(x => x.PlayerId == player.Id && x.Year == playerReceivingStats.Year).FirstOrDefaultAsync();
                        if (playerSeason == null)
                        {
                            playerSeason = new PlayerSeason()
                            {
                                GamesPlayed = int.Parse(Games),
                                Year = new DateTime(int.Parse(year), 1, 1),
                                IsCollegeSeason = true,
                                PlayerId = player.Id,
                            };
                            _context.Add(playerSeason);
                        }
                        

                        var playerPlaysForTeam = await _context.PlayerPlaysForTeam.Where(x => x.PlayerId == player.Id && x.TeamId == team.Id && x.EffectiveDate == new DateTime(int.Parse(year), 1, 1)).FirstOrDefaultAsync();

                        if (playerPlaysForTeam == null)
                        {
                            playerPlaysForTeam = new PlayerPlaysForTeam()
                            {
                                PlayerName = Name,
                                EffectiveDate = new DateTime(int.Parse(year), 1, 1),
                                PlayerId = player.Id,
                                TeamId = team.Id,
                            };

                            _context.Add(playerPlaysForTeam);
                        }
                        await _context.SaveChangesAsync();
                    }
                }
            }
        }

        public async Task AddPlayerClusters(string path)
        {
            string[] files = Directory.GetFiles(string.Format("{0}", path), "*.csv");
            foreach (string file in files)
            {
                if (!File.Exists(file))
                {
                    throw new Exception("File does not exist: " + file);
                }
                if (!file.Contains("meanshift"))
                {
                    continue;
                }
                using (TextFieldParser csvParser = new TextFieldParser(file))
                {
                    csvParser.SetDelimiters(new string[] { "," });
                    csvParser.HasFieldsEnclosedInQuotes = false;
                    while (!csvParser.EndOfData)
                    {
                        string[] fields = csvParser.ReadFields();
                        int playerId = int.Parse(fields[0]);
                        int cluster = int.Parse(fields[2]);

                        var player = await _context.Players.FirstOrDefaultAsync(x => x.Id == playerId);

                        if (player != null)
                        {
                            player.WRClusterMeanshift = cluster;

                            await _context.SaveChangesAsync();
                        }

                    }
                }
                break; // Only should be one file at a time
            }
        }

        public async Task ProcessNFLReceiverDataAndSaveToFile(string path)
        {
            string[] files = Directory.GetFiles(path, "*.csv");
            foreach (string file in files)
            {
                if (!File.Exists(file))
                {
                    throw new Exception("File does not exist: " + file);
                }
                var year = file.Split('\\').Last().Substring(0, 4);
                if  (int.Parse(year) > 2012 || int.Parse(year) < 1992)
                {
                    continue;
                }
                using (TextFieldParser csvParser = new TextFieldParser(file))
                {
                    csvParser.SetDelimiters(new string[] { "," });
                    csvParser.HasFieldsEnclosedInQuotes = false;
                    csvParser.ReadLine();
                    string Targets = string.Empty;
                    string Receptions = string.Empty;
                    string CatchPec = string.Empty;
                    string Yards = string.Empty;
                    string YardsPerReception = string.Empty;
                    string TDs = string.Empty;
                    string FirstDowns = string.Empty;
                    string SuccessPec = string.Empty;
                    string Long = string.Empty;
                    string YardsPerTarget = string.Empty;
                    string ReceptionsPerGame = string.Empty;
                    string YardsPerGame = string.Empty;
                    string Fumbles = string.Empty;
                    while (!csvParser.EndOfData)
                    {
                        string[] fields = csvParser.ReadFields();
                        string Name = fields[1].Trim(new char[] { '*', '+' });
                        string TeamName = fields[2];
                        string Age = fields[3];
                        string Position = fields[4];
                        string GamesPlayed = fields[5];
                        string GamesStarted = fields[6];

                        if (fields.Length == 17)
                        {
                            Receptions = fields[7];
                            Yards = fields[8];
                            YardsPerReception = fields[9];
                            TDs = fields[10];
                            Long = fields[11];
                            YardsPerTarget = fields[12];
                            ReceptionsPerGame = fields[13];
                            YardsPerGame = fields[14];
                            Fumbles = fields[15];
                        }
                        else if (fields.Length == 19)
                        {
                            Targets = fields[7];
                            Receptions = fields[8];
                            CatchPec = fields[9].Trim('%');
                            Yards = fields[10];
                            YardsPerReception = fields[11];
                            TDs = fields[12];
                            Long = fields[13];
                            YardsPerTarget = fields[14];
                            ReceptionsPerGame = fields[15];
                            YardsPerGame = fields[16];
                            Fumbles = fields[17];
                        }
                        else
                        {      
                            Targets = fields[7];
                            Receptions = fields[8];
                            CatchPec = fields[9].Trim('%');
                            Yards = fields[10];
                            YardsPerReception = fields[11];
                            TDs = fields[12];
                            FirstDowns = fields[13];
                            SuccessPec = fields[14].Trim('%');
                            Long = fields[15];
                            YardsPerTarget = fields[16];
                            ReceptionsPerGame = fields[17];
                            YardsPerGame = fields[18];
                            Fumbles = fields[19];
                        }

                        var player = await _context.Players.Where(x => x.Name == Name).FirstOrDefaultAsync();

                        if (player == null)
                        {
                            player = await CreateNewNFLPlayer(Name, GamesPlayed, GamesStarted, year, Age, Position);
                        }
                        if (player.Position == null)
                        {
                            player.Position = Position;
                            await _context.SaveChangesAsync();
                        }

                        var team = await _context.Teams.Where(x => x.TeamName == TeamName).FirstOrDefaultAsync();
                        if (team == null)
                        {
                            team = await CreateNewTeam(TeamName);
                        }

                        var season = await _context.PlayerSeason.Where(x => x.PlayerId == player.Id && x.Year.Year.ToString() == year).FirstOrDefaultAsync();
                        if (season != null)
                        {
                            season.IsCollegeSeason = false;
                            season.PlayerAge = int.Parse(Age);
                            season.GamesPlayed = int.Parse(GamesPlayed);
                            season.GamesStarted = int.Parse(GamesStarted);
                        }
                        else
                        {
                            season = new PlayerSeason();
                            season.IsCollegeSeason = false;
                            season.PlayerAge = int.Parse(Age);
                            season.GamesPlayed = int.Parse(GamesPlayed);
                            season.GamesStarted = int.Parse(GamesStarted);
                            season.Year = new DateTime(int.Parse(year), 1, 1);
                            season.PlayerId = player.Id;
                            _context.Add(season);
                        }


                        var seasonReceivingStats = await _context.PlayerReceivingStats.Where(x => x.PlayerId == player.Id && x.Year == season.Year).FirstOrDefaultAsync();
                        if (seasonReceivingStats == null)
                        {
                            try
                            {
                                seasonReceivingStats = new PlayerReceivingStats()
                                {
                                    PlayerId = player.Id,
                                    Year = season.Year,
                                    Targets = string.IsNullOrEmpty(Targets) ? 0 : int.Parse(Targets),
                                    Receptions = string.IsNullOrEmpty(Receptions) ? 0 : int.Parse(Receptions),
                                    Yards = string.IsNullOrEmpty(Yards) ? 0 : int.Parse(Yards),
                                    YardsPerReception = string.IsNullOrEmpty(YardsPerReception) ? 0 : decimal.Parse(YardsPerReception),
                                    YardsPerGame = string.IsNullOrEmpty(YardsPerGame) ? 0 : decimal.Parse(YardsPerGame),
                                    YardsPerTarget = string.IsNullOrEmpty(YardsPerTarget) ? 0 : decimal.Parse(YardsPerTarget),
                                    TDs = string.IsNullOrEmpty(TDs) ? 0 : int.Parse(TDs),
                                    Fumbles = string.IsNullOrEmpty(Fumbles) ? 0 : int.Parse(Fumbles),
                                    CatchPercentage = string.IsNullOrEmpty(CatchPec) ? 0 : decimal.Parse(CatchPec),
                                    FirstDownsReceiving = string.IsNullOrEmpty(FirstDowns) ? 0 : int.Parse(FirstDowns),
                                    LongestReception = string.IsNullOrEmpty(Long) ? 0 : int.Parse(Long),
                                    ReceivingSuccessRate = string.IsNullOrEmpty(SuccessPec) ? 0 : decimal.Parse(SuccessPec),
                                    ReceptionsPerGame = string.IsNullOrEmpty(ReceptionsPerGame) ? 0 : decimal.Parse(ReceptionsPerGame),
                                };
                            } catch (Exception ex)
                            {
                                _logger.LogError("Exception!");
                                throw;
                            }

                            _context.Add(seasonReceivingStats);
                        }
                        var playerPlaysForTeam = await _context.PlayerPlaysForTeam.Where(x => x.PlayerId == player.Id && x.TeamId == team.Id && x.EffectiveDate == new DateTime(int.Parse(year), 1, 1)).FirstOrDefaultAsync();
                        if (playerPlaysForTeam == null)
                        {
                            playerPlaysForTeam = new PlayerPlaysForTeam()
                            {
                                PlayerName = Name,
                                EffectiveDate = new DateTime(int.Parse(year), 1, 1),
                                PlayerId = player.Id,
                                TeamId = team.Id,
                            };

                            _context.Add(playerPlaysForTeam);
                        }
                        await _context.SaveChangesAsync();
                            
                        
                    }
                }
            }
        }


        public async Task<Team> CreateNewTeam(string teamName)
        {
            Team team = new Team() {
                TeamName = teamName,
            };


            _context.Add(team);

            await _context.SaveChangesAsync();

            return await _context.Teams.FirstAsync(x => x.TeamName == teamName);
        }

        public async Task<Player> CreateNewCollegePlayer(string Name, string Games, string year, string pfrId)
        {
            if (string.IsNullOrEmpty(Games))
            {
                Games = "0";
            }
            Player player = new Player()
            {
                Name = Name,
                ProFootballReferenceId = pfrId,
                Seasons = new List<PlayerSeason>()
                            {
                                new PlayerSeason()
                                {
                                    GamesPlayed = int.Parse(Games),
                                    Year = new DateTime(int.Parse(year), 1, 1),
                                    IsCollegeSeason = true,
                                }
                            },
            };

            
            _context.Add(player);

            await _context.SaveChangesAsync();

            return await _context.Players.FirstAsync(x => x.Name == Name);
        }

        private async Task<Player> CreateNewNFLPlayer(string name, string gamesPlayed, string gamesStarted, string year, string age, string position)
        {
            Player player = new Player()
            {
                Name = name,
                Position = string.IsNullOrEmpty(position) ? string.Empty : position.Length == 1 ? position : position.Substring(0,2),
                Seasons = new List<PlayerSeason>()
                            {
                                new PlayerSeason()
                                {
                                    GamesPlayed = int.Parse(gamesPlayed),
                                    GamesStarted = int.Parse(gamesStarted),
                                    PlayerAge = int.Parse(age),
                                    Year = new DateTime(int.Parse(year), 1, 1),
                                    IsCollegeSeason = true,
                                }
                            },
            };


            _context.Add(player);
            try
            {
                await _context.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                throw;
            }

            return await _context.Players.FirstAsync(x => x.Name == name);
        }

        public async Task LoadCollegeApiDataToDataBase()
        {
            //var playerIds = (await _context.Players.Where(x =>x.CFDBPlayerId == null).Select(x => x.Id).ToListAsync()).OrderBy(x => x).ToList();
            var playerIds = (await _context.PlayerSeason.Where(x => x.CFDBSeasonAdded != true).Select(x => x.PlayerId).Distinct().ToListAsync()).OrderBy(x => x).ToList();
            int playerIteration = 0;
            foreach(var playerId in playerIds)
            {
                playerIteration++;
                if (playerIteration % 500 == 0)
                {
                    _logger.LogInformation("Processing Iteration: " + playerIteration.ToString());
                }
                var player = await _context.Players.FirstOrDefaultAsync(x => x.Id == playerId);

                if (player == null)
                {
                    continue; // Case when multiple players returned from 
                }
                if (player.CFDBPlayerId == null)
                {
                    var playerSearchResults = await GetCollegePlayerMetaData(player.Name, playerId);

                    if (playerSearchResults == null)
                    {
                        continue; // Unable to find any matching players 
                    }

                    await UpdatePlayerEntityWithSearchResults(player, playerSearchResults);
                }

                var receivingStats = await _context.PlayerReceivingStats.Where(x => x.PlayerId == playerId).ToListAsync();
                foreach (var season in receivingStats)
                {
                    var playerSeason = await _context.PlayerSeason.Where(x => x.PlayerId == playerId && x.Year == season.Year).FirstOrDefaultAsync();

                    if (playerSeason == null)
                    {
                        _logger.LogWarning(string.Format("Player Season Not Found: {0}, {1}, {2}", playerId, player.Name, season.Year));
                        continue;
                    }

                    if (playerSeason.CFDBSeasonAdded != true)
                    {
                        var usage = await GetCollegePlayerUseageByPlayerIdAndYear(player.CFDBPlayerId.Value, season.Year.Year.ToString());
                        var playerPredictedPointsAdded = await GetCollegePlayerPredictedPointsAddedByPlayerIdAndYear(player.CFDBPlayerId.Value, season.Year.Year.ToString());
                        await UpdatePlayerUseageToPlayerSeason(usage, playerPredictedPointsAdded, season);

                        playerSeason.CFDBSeasonAdded = true;

                        await _context.SaveChangesAsync();
                    }                 
                }
            }
        }

        private async Task UpdatePlayerUseageToPlayerSeason(PlayerUsageDto useage, PlayerPredictedPointsAddedDto playerPredictedPointsAdded, PlayerReceivingStats season)
        {
            if (useage != null && useage.Usage != null)
            {
                season.OverallUseage = useage.Usage.Overall;
                season.PassingDownsUseage = useage.Usage.Pass;
                season.OverallUseage = useage.Usage.Rush;
                season.FirstDownUseage = useage.Usage.FirstDown;
                season.SecondDownUseage = useage.Usage.SecondDown;
                season.ThirdDownUseage = useage.Usage.ThirdDown;
                season.PassingDownsUseage = useage.Usage.PassingDowns;
                season.StandardDownsUseage = useage.Usage.StandardDowns;
            }
 
            if (playerPredictedPointsAdded != null && playerPredictedPointsAdded.AveragePPA != null)
            {
                season.AveragePPAAll = playerPredictedPointsAdded.AveragePPA.All;
                season.AveragePPAPass = playerPredictedPointsAdded.AveragePPA.Pass;
                season.AveragePPARush = playerPredictedPointsAdded.AveragePPA.Rush;
                season.AveragePPAFirstDown = playerPredictedPointsAdded.AveragePPA.FirstDown;
                season.AveragePPASecondDown = playerPredictedPointsAdded.AveragePPA.SecondDown;
                season.AveragePPAThirdDown = playerPredictedPointsAdded.AveragePPA.ThirdDown;
                season.AveragePPAPassingDowns = playerPredictedPointsAdded.AveragePPA.PassingDowns;
                season.AveragePPAStandardDowns = playerPredictedPointsAdded.AveragePPA.StandardDowns;
            }

            if (playerPredictedPointsAdded != null && playerPredictedPointsAdded.TotalPPA != null)
            {
                season.TotalPPAAll = playerPredictedPointsAdded.TotalPPA.All;
                season.TotalPPAPass = playerPredictedPointsAdded.TotalPPA.Pass;
                season.TotalPPARush = playerPredictedPointsAdded.TotalPPA.Rush;
                season.TotalPPAFirstDown = playerPredictedPointsAdded.TotalPPA.FirstDown;
                season.TotalPPASecondDown = playerPredictedPointsAdded.TotalPPA.SecondDown;
                season.TotalPPAThirdDown = playerPredictedPointsAdded.TotalPPA.ThirdDown;
                season.TotalPPAPassingDowns = playerPredictedPointsAdded.TotalPPA.PassingDowns;
                season.TotalPPAStandardDowns = playerPredictedPointsAdded.TotalPPA.StandardDowns;
            }

            await _context.SaveChangesAsync();
        }

        public async Task CalculateYearTurnedPro()
        {
            int i = 0;

            var playersTurnedPro = await _context.Players.Where(x => x.YearTurnedPro == null).ToListAsync();

            foreach(var player in playersTurnedPro)
            {
                if (i++ % 500 == 0)
                {
                    _logger.LogInformation("Iteration: "+ i); 
                }
                var playerSeasons = (await _context.PlayerSeason.Where(x => x.PlayerId == player.Id).ToListAsync()).OrderBy(x =>x.Year).ToList();

                var yearTurnedPro = playerSeasons.FirstOrDefault(x => x.IsCollegeSeason != true);
                if (yearTurnedPro != null)
                {
                    var seasonIndex = playerSeasons.IndexOf(yearTurnedPro);
                    if (seasonIndex != 0)
                    {
                        player.YearTurnedPro = yearTurnedPro.Year;

                        await _context.SaveChangesAsync();
                    }

                }
            }
        }

        public async Task AddPlayerAges()
        {
            int i = 0;

            var players = await _context.Players.ToListAsync();

            foreach (var player in players)
            {
                if (i++ % 500 == 0)
                {
                    _logger.LogInformation("Iteration: " + i);
                }
                var playerSeasons = (await _context.PlayerSeason.Where(x => x.PlayerId == player.Id ).ToListAsync()).OrderByDescending(x => x.Year).ToList();

                var firstSeason = playerSeasons.LastOrDefault(x => x.PlayerAge != 0 && x.PlayerAge != null);
                if (firstSeason != null)
                {
                    var index = playerSeasons.IndexOf(firstSeason);
                    if (index >= 0)
                    {
                        var age = firstSeason.PlayerAge;
                        while (++index < playerSeasons.Count)
                        {
                            var season = playerSeasons.ElementAt(index);
                            season.PlayerAge = --age;
                        }
                        await _context.SaveChangesAsync();
                    }

                }
            }
            }

        private async Task<PlayerPredictedPointsAddedDto> GetCollegePlayerPredictedPointsAddedByPlayerIdAndYear(int playerId, string year)
        {
            var query = string.Format("ppa/players/season?year={0}&playerId={1}", year, playerId);
            HttpResponseMessage response = await _client.GetAsync(query);

            response.EnsureSuccessStatusCode();
            var jsonResponse = await response.Content.ReadAsStringAsync();

            var jsonSerializerSettings = new JsonSerializerSettings();
            jsonSerializerSettings.MissingMemberHandling = MissingMemberHandling.Ignore;

            return JsonConvert.DeserializeObject<List<PlayerPredictedPointsAddedDto>>(jsonResponse, jsonSerializerSettings).FirstOrDefault();
        }

        private async Task<PlayerUsageDto> GetCollegePlayerUseageByPlayerIdAndYear(int playerId, string year)
        {
            var query = string.Format("player/usage?playerId={0}&year={1}", playerId, year);
            HttpResponseMessage response = await _client.GetAsync(query);

            response.EnsureSuccessStatusCode();
            var jsonResponse = await response.Content.ReadAsStringAsync();

            var jsonSerializerSettings = new JsonSerializerSettings();
            jsonSerializerSettings.MissingMemberHandling = MissingMemberHandling.Ignore;

            return JsonConvert.DeserializeObject<List<PlayerUsageDto>>(jsonResponse, jsonSerializerSettings).FirstOrDefault();
        }

        public async Task<PlayerSearchDto> GetCollegePlayerMetaData(string playerName, int playerId)
        {
            var query = string.Format("player/search?searchTerm={0}", Uri.EscapeDataString(playerName));
            HttpResponseMessage  response = await _client.GetAsync(query);
            try
            {
                response.EnsureSuccessStatusCode();
                var jsonResponse = await response.Content.ReadAsStringAsync();

                var jsonSerializerSettings = new JsonSerializerSettings();
                jsonSerializerSettings.MissingMemberHandling = MissingMemberHandling.Ignore;

                var players = JsonConvert.DeserializeObject<List<PlayerSearchDto>>(jsonResponse, jsonSerializerSettings);

                if (players.Count > 1)
                {
                    _logger.LogWarning(string.Format("Multiple player metadata fetched with Name: {0}, PlayerId: {1}", playerName, playerId));

                    var playerPlaysForTeams = await _context.PlayerPlaysForTeam.Where(x => x.PlayerId == playerId).Include(x => x.Team).AsNoTracking().ToListAsync();
                    foreach (var player in players)
                    {
                        if (playerPlaysForTeams.Any(x => x.Team.TeamName.Contains(player.Team)) || playerPlaysForTeams.Any(x =>player.Team.Contains(x.Team.TeamName)))
                        {
                            return player;
                        }
                    }

                    return null;    
                }

                return players.FirstOrDefault();

            } catch (HttpRequestException httpException)
            {
                _logger.LogError("Error encountered with http GET request " + query);
                return null;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex.Message);
                throw ex;
            }
        }

        private async Task UpdatePlayerEntityWithSearchResults(Player player, PlayerSearchDto playerSearchResults)
        {
            player.Position = playerSearchResults.Position;
            player.Height = playerSearchResults.Height;
            player.Weight = playerSearchResults.Weight;
            player.CFDBPlayerId = playerSearchResults.Id;

            await _context.SaveChangesAsync();
        }
    }
}
