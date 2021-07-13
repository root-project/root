const { Octokit } = require("octokit");

const token = process.env.GITHUB_TOKEN;

async function check_issue(owner, repo, issue) {
  const octokit = new Octokit({ auth: token });

  const { data } = await octokit.rest.issues.listEvents({
    owner: owner,
    repo: repo,
    issue_number: issue,
  });
  const added_to_project = data.filter(e => e["event"] == "added_to_project");
  const removed_from_project = data.filter(e => e["event"] == "removed_from_project");
  const projects = added_to_project.length - removed_from_project.length;

  if (projects > 0) {
    console.log("It appears this issue already has at least one project.");
  } else {
    console.log("It appears this issue has no project...");

    const { data } = await octokit.rest.issues.get({
      owner: owner,
      repo: repo,
      issue_number: issue,
    });
    const assignees = data["assignees"].map(user => user.login);
    const closed_by = data["closed_by"]["login"];

    body = "Hi @" + closed_by;
    for (let i = 0; i < assignees.length; i++) {
      if (assignees[i] == closed_by) {
        continue;
      }
      body += ", @" + assignees[i];
    }
    body += ",\n\n";
    body += "It appears this issue wasn't added to a project before closing. ";
    body += "Please add upcoming versions that will include the fix, or 'not applicable' otherwise.";
    body += "\n\nSincerly,\n:robot:\n";

    await octokit.rest.issues.createComment({
      owner: owner,
      repo: repo,
      issue_number: issue,
      body: body,
    });
  }
}

const [owner, repo] = process.env.GITHUB_REPOSITORY.split("/");
const issue = process.argv[2];
check_issue(owner, repo, issue);
